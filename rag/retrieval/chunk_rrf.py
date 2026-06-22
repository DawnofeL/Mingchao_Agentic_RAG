"""Ming_Dynasty chunk 检索器：dense + sparse + RRF + reranker。

本模块是 Vector Mode 和 Agentic Mode chunk 分支共用的检索核心，不含任何 LLM 调用。
HNSW / 倒排索引的索引结构和搜索算法本身不在本模块里实现，本模块只通过 pymilvus
SDK 把向量和搜索参数发给 Milvus 服务端（或 Milvus Lite 进程），由 Milvus 底层
（基于 FAISS 一类向量索引库）完成真正的索引和检索，这里只负责发请求、收结果、
转格式，以及拿到两路结果之后做 RRF 融合和精排。

本模块定义以下函数，调用关系如下：

    Inject_Models        外部注入已加载模型，避免重复初始化，供 Agentic 模式与 Vector 模式共享同一实例。
    Preload_Models       预热编码器、精排模型和 Milvus Collection，notebook init cell 调用一次即可。

    Search_Chunks        主函数，完整执行双路检索流水线，返回 top-k chunk 列表。
        └── _Encode_Query       单次 BGE-M3 前向传播，同时拿到 dense 向量和 sparse 权重字典。
        └── _Search_One_Path    调用两次，对 dense 和 sparse 各执行一路 Milvus 向量检索。
        └── _RRF_Core           对两路结果做 RRF 融合，返回按分值降序的候选列表。
        └── reranker.compute_score    FlagReranker 精排，按精排分降序取 top_k。

    懒加载内部函数（进程级单例，只初始化一次）：
        _Get_Encoder     加载 BGE-M3 编码器。
        _Get_Reranker    加载 FlagReranker 精排模型。
        _Get_Collection  连接 Milvus 并加载 chunk collection。

使用方式（notebook init cell）：
    from rag.retrieval.chunk_rrf import Preload_Models
    Preload_Models()   # 提前预热，后续调用零启动开销
"""

import logging
import threading

import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pymilvus import Collection, connections

# FlagEmbedding tokenizer 内部触发的 HuggingFace UserWarning 与功能无关，统一过滤保持日志整洁
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from rag.config.settings import (
    BGE_MODEL_PATH,
    CHUNK_CANDIDATES,
    CHUNK_DENSE_FIELD,
    CHUNK_COLLECTION,
    CHUNK_OUTPUT_FIELDS,
    CHUNK_PRIMARY_KEY,
    CHUNK_SPARSE_FIELD,
    CHUNK_TEXT_FIELD,
    CHUNK_TOP_K,
    DENSE_METRIC_TYPE,
    HNSW_EF,
    MILVUS_DB,
    MILVUS_HOST,
    MILVUS_MODE,
    MILVUS_LITE_PATH,
    MILVUS_PORT,
    RERANKER_LOW_SCORE_THRESHOLD,
    RERANKER_MODEL_PATH,
    RRF_K,
    SPARSE_DROP_RATIO,
)


# ── Milvus Lite 适配层 ────────────────────────────────────────────────────────
# 让 MilvusClient 的搜索结果伪装成 pymilvus Collection.search() 的格式，
# 使 _Search_One_Path 无需感知底层是 Docker 还是 Lite。

class _FakeEntity:
    """把 dict 包一层，提供与 pymilvus hitEntity 相同的 .get() 接口。"""

    def __init__(self, d: dict) -> None:
        self._d = d

    def get(self, key: str):
        return self._d.get(key)


class _FakeHit:
    """模拟 pymilvus Hit 对象，暴露 .distance 和 .entity。"""

    def __init__(self, distance: float, entity: dict) -> None:
        self.distance = distance
        self.entity   = _FakeEntity(entity)


class _LiteCollection:
    """MilvusClient 的薄包装，search() 接口与 pymilvus Collection 保持一致。"""

    def __init__(self, client, name: str) -> None:
        self._client = client
        self._name   = name

    def search(
        self,
        data:          list,
        anns_field:    str,
        param:         dict,
        limit:         int,
        output_fields: list[str],
    ) -> list[list[_FakeHit]]:
        # MilvusClient 用 search_params 而不是 param，其余参数相同
        raw = self._client.search(
            collection_name = self._name,
            data            = data,
            anns_field      = anns_field,
            search_params   = param,
            limit           = limit,
            output_fields   = output_fields,
        )
        return [
            [_FakeHit(h.distance, h.entity) for h in hits]
            for hits in raw
        ]


# ── 进程级单例（懒加载，只初始化一次）───────────────────────────────────────

_bge_encoder:      BGEM3FlagModel | None            = None
_reranker:         FlagReranker | None               = None
_chunk_collection: Collection | _LiteCollection | None = None

# encoder 和 reranker 的 model.half() 不是线程安全的，并发时必须串行
_model_lock = threading.Lock()


# ── 外部注入 ──────────────────────────────────────────────────────────────────

def Inject_Models(
    encoder:    BGEM3FlagModel,
    reranker:   FlagReranker,
    collection: Collection,
) -> None:
    """将外部已加载的模型和 collection 注入模块单例，避免重复初始化。

    Agentic Mode 与 Vector Mode 共享同一批资源时调用此函数。
    注入后后续所有 Search_Chunks 调用都直接复用这些实例。

    Args:
        encoder:    已加载的 BGEM3FlagModel 编码器实例。
        reranker:   已加载的 FlagReranker 精排模型实例。
        collection: 已加载的 Milvus Collection 实例。
    """

    global _bge_encoder, _reranker, _chunk_collection
    _bge_encoder      = encoder
    _reranker         = reranker
    _chunk_collection = collection
    print("[chunk_rrf] 模型注入完成，直接复用已加载实例。")


# ── 懒加载（进程内只初始化一次）──────────────────────────────────────────────

def _Select_Device() -> str:
    """选择推理设备，优先级 CUDA > Apple MPS > CPU，逐级退行。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _Get_Encoder() -> BGEM3FlagModel:
    """返回 BGE-M3 编码器单例，首次调用时加载，后续直接复用。"""

    global _bge_encoder
    if _bge_encoder is None:
        device       = _Select_Device()
        _bge_encoder = BGEM3FlagModel(BGE_MODEL_PATH, use_fp16 = (device == "cuda"), devices = device)
        print(f"[chunk_rrf] BGE-M3 加载完成，设备={device}")
    return _bge_encoder


def _Get_Reranker() -> FlagReranker:
    """返回 FlagReranker 精排模型单例，首次调用时加载，后续直接复用。"""

    global _reranker
    if _reranker is None:
        device    = _Select_Device()
        _reranker = FlagReranker(RERANKER_MODEL_PATH, use_fp16 = (device == "cuda"), devices = device)
        print(f"[chunk_rrf] Reranker 加载完成，设备={device}")
    return _reranker


def _Get_Collection() -> Collection | _LiteCollection:
    """连接 Milvus 并返回 chunk collection 单例，进程内只初始化一次。

    MILVUS_MODE = "docker"：沿用原有 pymilvus connections + Collection 路径，行为不变。
    MILVUS_MODE = "lite"  ：使用本地 MilvusClient，首次运行自动建表并导入数据。
    """

    global _chunk_collection
    if _chunk_collection is not None:
        return _chunk_collection

    if MILVUS_MODE == "lite":
        from pymilvus import MilvusClient
        from rag.retrieval.milvus_lite_setup import ensure_lite_db

        # 首次运行建表导入，已存在则直接跳过
        ensure_lite_db()
        client = MilvusClient(uri=MILVUS_LITE_PATH)

        # 新进程打开 db 后 collection 处于 released，必须先 load 才能 search
        client.load_collection(CHUNK_COLLECTION)
        _chunk_collection = _LiteCollection(client, CHUNK_COLLECTION)
    else:
        connections.connect(
            alias   = "default",
            host    = MILVUS_HOST,
            port    = MILVUS_PORT,
            db_name = MILVUS_DB,
        )
        col = Collection(CHUNK_COLLECTION)
        col.load()
        _chunk_collection = col

    print(f"[chunk_rrf] Collection '{CHUNK_COLLECTION}' 加载完成 (mode={MILVUS_MODE})")
    return _chunk_collection


def Preload_Models() -> None:
    """预热编码器、精排模型和 Milvus Collection。

    在 notebook init cell 调用一次，后续所有 Search_Chunks 调用零启动开销。
    """

    _Get_Encoder()
    _Get_Reranker()
    _Get_Collection()
    print("[chunk_rrf] 所有资源预热完成。")


# ── 内部检索函数 ──────────────────────────────────────────────────────────────

def _Encode_Query(query: str) -> tuple[list[float], dict[int, float]]:
    """单次 BGE-M3 前向传播，同时拿到 dense 向量和 sparse 权重字典。

    用 return_dense + return_sparse 合并调用，避免两次前向传播的额外开销。

    Args:
        query: 检索问题文本。
    Returns:
        (dense_vec, sparse_vec)。
        dense_vec 是长度 1024 的 float 列表；
        sparse_vec 是 {token_id: weight} 字典。
    """

    encoder = _Get_Encoder()

    # 拿锁再调用 encoder.encode，跟 Search_Chunks 里 reranker.compute_score 抢的是
    # 同一把 _model_lock，确保两个模型任何时候都只有一个线程在跑前向传播
    with _model_lock:
        output = encoder.encode(
            [query],
            return_dense  = True,
            return_sparse = True,
        )

    dense_vec  = output["dense_vecs"][0].tolist()
    sparse_vec = {int(k): float(v) for k, v in output["lexical_weights"][0].items()}
    return dense_vec, sparse_vec


def _Search_One_Path(
    collection:   Collection,
    query_vector,
    anns_field:   str,
    search_param: dict,
    top_k:        int,
) -> list[tuple]:
    """对一个向量字段执行一次 Milvus 检索，dense 和 sparse 各调用一次，互不感知对方。

    Search_Chunks 里会拿同一个 collection 分别传 dense 向量配 COSINE，
    再传 sparse 向量配 IP，调两次这个函数，分别拿到两条独立排序的命中列表。
    返回值统一成 (chunk_id, rank, distance, entity_dict) 元组，而不是直接
    返回 Milvus 原始 hit 对象，是因为 _RRF_Core 融合两路结果时只关心
    每个 chunk_id 在这一路里排第几（rank），不关心具体 distance 数值，
    统一格式之后 _RRF_Core 才能不区分是哪一路直接拼起来算分。

    Args:
        collection:   Milvus Collection 实例。
        query_vector: dense 向量（list[float]）或 sparse 向量（dict）。
        anns_field:   向量字段名，dense 用 "embedding"，sparse 用 "sparse_embedding"。
        search_param: Milvus 搜索参数字典，例如 {"metric_type": "COSINE", "params": {"ef": 64}}。
        top_k:        本路返回条数。
    Returns:
        list of (chunk_id, rank, distance, entity_dict)。
    """

    # collection.search() 是 pymilvus SDK 提供的方法，这里只是把要查的向量、
    # 查哪个字段（anns_field）、用什么相似度算法和索引参数（search_param）打包
    # 发给 Milvus 服务端（或 Milvus Lite 进程）。HNSW 的多层图结构和图遍历算法
    result = collection.search(
        data          = [query_vector],
        anns_field    = anns_field,
        param         = search_param,
        limit         = top_k,
        output_fields = CHUNK_OUTPUT_FIELDS,
    )

    hits = []

    # 把 Milvus 返回的 hit 对象转换成 (chunk_id, rank, distance, entity_dict)
    for rank, hit in enumerate(result[0], 1):
        entity = {f: hit.entity.get(f) for f in CHUNK_OUTPUT_FIELDS}
        hits.append((hit.entity.get(CHUNK_PRIMARY_KEY), rank, float(hit.distance), entity))

    return hits


def _RRF_Core(hit_lists: list[list[tuple]], top_k: int) -> list[dict]:
    """对多路候选做 RRF 融合，返回按 rrf_score 降序的 top_k 列表。

    RRF 公式：score(d) = Σ 1 / (k + rank_i(d))，k 取 settings.RRF_K。
    未出现在某路结果中的文档，该路 rank 视为无穷大（score 贡献接近 0）。

    Args:
        hit_lists: 各路检索结果，每路是 _Search_One_Path 返回的元组列表。
        top_k:     融合后返回条数。
    Returns:
        list[dict]，把传进来的多路结果合并成单一一份列表，按 rrf_score 降序排好，
        每条 dict 是原始 entity 字段（卷号、正文等）加上新算出的 rrf_score 和
        rrf_rank（在这份合并列表里排第几）。每路各自原来的 rank 只在算分时用过，
        合并完之后就不在返回值里了，不会保留"dense 第几名 / sparse 第几名"这种信息。
    """

    entity_cache: dict    = {}
    rank_maps: list[dict] = []

    # hit_lists 是 [dense_hits, sparse_hits]，每个元素（这里命名 hits）就是
    # _Search_One_Path 的返回值，一条 hits 对应一路检索结果。这个循环把每路
    # hits 转成一个 rank_map（chunk_id → 在这一路里排第几），所有路的
    # rank_map 最后存进 rank_maps，供下面算 RRF 分数时使用
    for hits in hit_lists:
        rank_map: dict = {}

        # hits 里每条是 (chunk_id, rank, distance, entity)，
        # 这里只要 chunk_id 和 rank，distance 不参与 RRF 计算所以丢掉
        for cid, rank, _, entity in hits:
            rank_map[cid]     = rank
            entity_cache[cid] = entity

        rank_maps.append(rank_map)

    # 两路的 chunk_id 取并集，只要任意一路命中过就要参与算分
    all_ids = set().union(*(m.keys() for m in rank_maps))

    # 对每个 chunk_id，把它在每一路 rank_map 里的排名代入公式 1/(k+rank) 再求和；
    # 如果这个 chunk_id 在某一路里没出现，m.get(cid, 10**9) 给一个极大排名当占位，
    # 1/(k+极大值) 约等于 0，相当于这一路对总分没有实质贡献
    rrf_scores = {
        cid: sum(1.0 / (RRF_K + m.get(cid, 10 ** 9)) for m in rank_maps)
        for cid in all_ids
    }

    # 按 rrf_score 从高到低排序，只保留前 top_k 个 chunk_id
    sorted_ids = sorted(all_ids, key = lambda x: rrf_scores[x], reverse = True)[:top_k]

    merged = []
    for rank, cid in enumerate(sorted_ids, 1):
        if cid not in entity_cache:
            continue

        # 把这个 chunk 原本的字段（卷号、正文等）取出来，再把这一步算出的
        # rrf_score 和它在融合结果里的最终排名 rrf_rank 一起塞进同一个 dict
        row              = dict(entity_cache[cid])
        row["rrf_score"] = rrf_scores[cid]
        row["rrf_rank"]  = rank
        merged.append(row)
    return merged


# ── 对外主函数 ────────────────────────────────────────────────────────────────

def Search_Chunks(
    query:      str,
    top_k:      int = CHUNK_TOP_K,
    candidates: int = CHUNK_CANDIDATES,
) -> list[dict]:
    """BGE-M3 dense + sparse 双路检索 → RRF 融合 → reranker 精排 → top-k 返回。

    完整检索流水线：
        1. 单次 encode 同时拿 dense 向量和 sparse 权重。
        2. dense（COSINE+HNSW）和 sparse（IP）各取 candidates 条候选。
        3. RRF 融合两路候选，取 candidates 条。
        4. FlagReranker 对每条候选打分，按精排分降序取 top_k。
        5. top-1 精排分低于阈值时视为无相关内容，返回空列表。

    Args:
        query:      检索问题文本。
        top_k:      最终返回 chunk 数，默认 CHUNK_TOP_K（10）。
        candidates: RRF 前各路候选数，默认 CHUNK_CANDIDATES（30）。
    Returns:
        list[dict]，每条含 chunk_id / volume / chapter / section /
        chunk_index / chunk_total / content / rrf_score / rrf_rank / reranker_score。
        无相关 chunk 时返回空列表。
    """

    reranker   = _Get_Reranker()
    collection = _Get_Collection()

    # BGE-M3 一次前向传播能同时输出 dense 和 sparse 两种表示，所以只 encode
    # 一次，不分别调两次模型，省掉一次重复的前向传播开销
    dense_vec, sparse_vec = _Encode_Query(query)

    # dense 向量捕捉的是语义相似度，所以用 COSINE 距离 + HNSW 索引去搜
    dense_hits = _Search_One_Path(
        collection, dense_vec, CHUNK_DENSE_FIELD,
        {"metric_type": DENSE_METRIC_TYPE, "params": {"ef": HNSW_EF}},
        candidates,
    )

    # sparse 向量本质是关键词权重，找的是"用词相近"的内容，用内积匹配权重，
    # drop_ratio_search 把权重太低的虚词过滤掉，避免虚词干扰匹配结果
    sparse_hits = _Search_One_Path(
        collection, sparse_vec, CHUNK_SPARSE_FIELD,
        {"metric_type": "IP", "params": {"drop_ratio_search": SPARSE_DROP_RATIO}},
        candidates,
    )

    # 两路分数量级不同没法直接比，RRF 只按排名融合成一个候选池
    pool = _RRF_Core([dense_hits, sparse_hits], top_k = candidates)
    if not pool:
        return []

    # 组装 (query, chunk正文) 配对列表，交给 reranker 重新打真实相关性分
    pair_list = [[query, item[CHUNK_TEXT_FIELD]] for item in pool]

    # 同一把 _model_lock，跟 _Encode_Query 里锁 encoder.encode 是同一个目的，
    # 避免并发请求时两个线程同时调 model.half() 出问题
    with _model_lock:
        reranker_scores = reranker.compute_score(pair_list, normalize = True)

    # 只有一条候选时 compute_score 返回单个 float，转成列表方便下面 zip
    if isinstance(reranker_scores, float):
        reranker_scores = [reranker_scores]

    for item, score in zip(pool, reranker_scores):
        item["reranker_score"] = float(score)

    # 按 reranker 的真实相关性分重新排序，取最终 top_k
    result = sorted(pool, key = lambda x: x["reranker_score"], reverse = True)[:top_k]

    # 前面几步是强行排序，不保证真的相关，这里用分数兜底过滤掉不相关结果
    if result and result[0]["reranker_score"] < RERANKER_LOW_SCORE_THRESHOLD:
        print(
            f"[chunk_rrf] top-1 reranker_score={result[0]['reranker_score']:.4f} "
            f"< {RERANKER_LOW_SCORE_THRESHOLD}，无相关 chunk"
        )
        return []

    return result
