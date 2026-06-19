"""Ming_Dynasty chunk 检索器：dense + sparse + RRF + reranker。

本模块是 Vector Mode 和 Agentic Mode chunk 分支共用的检索核心，不含任何 LLM 调用。

本模块定义以下函数，调用关系如下：

    Inject_Models        外部注入已加载模型，避免重复初始化，供 Agentic 模式与 Vector 模式共享同一实例。
    Preload_Models       预热编码器、精排模型和 Milvus Collection，notebook init cell 调用一次即可。

    Search_Chunks        主函数，完整执行双路检索流水线，返回 top-k chunk 列表。
        └── _Encode_Query       单次 BGE-M3 前向传播，同时拿到 dense 向量和 sparse 权重字典。
        └── _Search_One_Path    × 2   对 dense 和 sparse 各执行一路 Milvus 向量检索。
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
_model_lock = threading.Lock()  # encoder 和 reranker 的 model.half() 不是线程安全的，并发时必须串行


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

        ensure_lite_db()  # 首次运行建表导入，已存在则直接跳过
        client = MilvusClient(uri=MILVUS_LITE_PATH)
        client.load_collection(CHUNK_COLLECTION)  # 新进程打开 db 后 collection 处于 released，必须先 load 才能 search
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
    """执行单路 Milvus 向量检索，返回标准元组列表供 RRF 使用。

    Args:
        collection:   Milvus Collection 实例。
        query_vector: dense 向量（list[float]）或 sparse 向量（dict）。
        anns_field:   向量字段名，dense 用 "embedding"，sparse 用 "sparse_embedding"。
        search_param: Milvus 搜索参数字典，例如 {"metric_type": "COSINE", "params": {"ef": 64}}。
        top_k:        本路返回条数。
    Returns:
        list of (chunk_id, rank, distance, entity_dict)。
    """

    result = collection.search(
        data          = [query_vector],
        anns_field    = anns_field,
        param         = search_param,
        limit         = top_k,
        output_fields = CHUNK_OUTPUT_FIELDS,
    )

    hits = []
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
        list[dict]，每条含原始 entity 字段 + rrf_score + rrf_rank。
    """

    entity_cache: dict    = {}
    rank_maps: list[dict] = []

    # 把每路结果拆成 rank_map（chunk_id → rank）和 entity_cache
    for hits in hit_lists:
        route_map: dict = {}
        for cid, rank, _, entity in hits:
            route_map[cid]    = rank
            entity_cache[cid] = entity
        rank_maps.append(route_map)

    # 计算所有出现过的 chunk_id 的 RRF 得分
    all_ids    = set().union(*(m.keys() for m in rank_maps))
    rrf_scores = {
        cid: sum(1.0 / (RRF_K + m.get(cid, 10 ** 9)) for m in rank_maps)
        for cid in all_ids
    }

    # 按分值降序取 top_k，附上 rrf_score 和 rrf_rank 字段
    sorted_ids = sorted(all_ids, key = lambda x: rrf_scores[x], reverse = True)[:top_k]

    merged = []
    for rank, cid in enumerate(sorted_ids, 1):
        if cid not in entity_cache:
            continue
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

    # 单次 encode，同时得到 dense 向量和 sparse 权重字典
    dense_vec, sparse_vec = _Encode_Query(query)

    # dense 路：COSINE 相似度，HNSW 索引
    dense_hits = _Search_One_Path(
        collection, dense_vec, CHUNK_DENSE_FIELD,
        {"metric_type": DENSE_METRIC_TYPE, "params": {"ef": HNSW_EF}},
        candidates,
    )

    # sparse 路：内积相似度（BGE-M3 稀疏向量标准配置），drop_ratio 过滤低权重 token
    sparse_hits = _Search_One_Path(
        collection, sparse_vec, CHUNK_SPARSE_FIELD,
        {"metric_type": "IP", "params": {"drop_ratio_search": SPARSE_DROP_RATIO}},
        candidates,
    )

    # RRF 融合两路候选，产出统一排名的候选池
    pool = _RRF_Core([dense_hits, sparse_hits], top_k = candidates)
    if not pool:
        return []

    # FlagReranker 对每对 (query, chunk_content) 打精排分
    pair_list = [[query, item[CHUNK_TEXT_FIELD]] for item in pool]
    with _model_lock:
        reranker_scores = reranker.compute_score(pair_list, normalize = True)

    # compute_score 在单条输入时返回 float 而非列表，统一转成列表处理
    if isinstance(reranker_scores, float):
        reranker_scores = [reranker_scores]

    for item, score in zip(pool, reranker_scores):
        item["reranker_score"] = float(score)

    # 按精排分降序取 top_k
    result = sorted(pool, key = lambda x: x["reranker_score"], reverse = True)[:top_k]

    # top-1 分数过低说明知识库内没有相关内容，直接返回空
    if result and result[0]["reranker_score"] < RERANKER_LOW_SCORE_THRESHOLD:
        print(
            f"[chunk_rrf] top-1 reranker_score={result[0]['reranker_score']:.4f} "
            f"< {RERANKER_LOW_SCORE_THRESHOLD}，无相关 chunk"
        )
        return []

    return result
