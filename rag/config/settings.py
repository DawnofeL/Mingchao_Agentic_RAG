"""全局配置。

本模块集中管理所有可调参数，按功能分为以下几组：
    Milvus 连接参数        — 主机、端口、数据库名。
    Collection 名称        — 各知识库对应的 Milvus collection。
    模型路径               — BGE-M3 编码器和 Reranker 的本地路径。
    检索数量               — 最终 top_k、RRF 候选池大小、RRF 阻尼系数。
    Dense / Sparse 搜索参数 — HNSW ef、drop_ratio_search 等超参数。
    Reranker 阈值          — 低分判定标准。
    Chunk 字段名           — Milvus collection 中各字段的实际名称。
路径策略：用 Path(__file__).resolve() 确保 WSL / Windows 双端都能正确解析。
"""

import os
from pathlib import Path

from langchain_openai import ChatOpenAI


# 项目根目录：settings.py → config → rag → RAG_Ming_Refine
_ROOT = Path(__file__).resolve().parents[2]


# ── Milvus 模式 ───────────────────────────────────────────────────────────────
# "docker" : 连接本地 Docker Milvus（默认，现有行为不变）
# "lite"   : 使用 Milvus Lite 本地文件，无需 Docker，适合开源分发

MILVUS_MODE = "lite"

# ── Milvus 连接（Docker 模式）────────────────────────────────────────────────

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_DB   = "SednaAI"

# ── Milvus Lite 路径（Lite 模式）─────────────────────────────────────────────

MILVUS_LITE_PATH      = str(_ROOT / "data" / "milvus_lite.db")
MILVUS_LITE_CHUNKS_JSON = str(_ROOT / "data" / "raw" / "mingchao_vectorized_1_661.json")


# ── Collection 名称 ───────────────────────────────────────────────────────────

CHUNK_COLLECTION = "Ming_Dynasty"

# Agentic Mode 后续使用，先注释占位：
# PEOPLE_COLLECTION   = "Ming_Dynasty_People"
# TIMELINE_COLLECTION = "Ming_Dynasty_Timeline"


# ── 模型路径（相对于项目根目录，自动解析为绝对路径）─────────────────────────

BGE_MODEL_PATH      = str(_ROOT / "model" / "BAAI_bge-m3")
RERANKER_MODEL_PATH = str(_ROOT / "model" / "BAAI_bge-reranker-v2-m3")


# ── 检索数量 ──────────────────────────────────────────────────────────────────

CHUNK_TOP_K      = 10   # reranker 后最终返回 chunk 数
CHUNK_CANDIDATES = 30   # dense / sparse 各路召回候选数，进入 RRF 的池子大小
RRF_K            = 60   # RRF 阻尼系数，越大排名差异越平滑


# ── Dense 搜索参数 ────────────────────────────────────────────────────────────

DENSE_METRIC_TYPE = "COSINE"   # 距离度量，与建索引时保持一致
HNSW_EF           = 64         # HNSW 搜索 ef，越大召回质量越高但越慢


# ── Sparse 搜索参数 ───────────────────────────────────────────────────────────

SPARSE_DROP_RATIO = 0.2   # drop_ratio_search：权重低于 max×0.2 的 token 不发给 Milvus，节省检索开销


# ── Reranker ──────────────────────────────────────────────────────────────────

RERANKER_LOW_SCORE_THRESHOLD = 0.05   # top-1 低于此值视为无相关内容，直接返回空列表


# ── Chunk 字段配置（与 Milvus schema 保持一致）───────────────────────────────

CHUNK_OUTPUT_FIELDS = [
    "chunk_id", "volume", "chapter", "section",
    "chunk_index", "chunk_total", "content",
]
CHUNK_TEXT_FIELD   = "content"
CHUNK_PRIMARY_KEY  = "chunk_id"
CHUNK_DENSE_FIELD  = "embedding"
CHUNK_SPARSE_FIELD = "sparse_embedding"


# ── LLM ──────────────────────────────────────────────────────────────────────

# 支持两家供应商：通义千问 Qwen（阿里云 DashScope）和 DeepSeek，接口都走 OpenAI 兼容格式。
# 每家的接口地址、默认模型、取 key 的环境变量、是否支持思考参数都登记在下面这张表里。
# 思考参数（enable_thinking）是 Qwen 自家的，DeepSeek 接口不认，所以只有 supports_thinking 的家才带。
# key 不落盘到任何项目文件，推荐运行时提供，优先级最高：
# 网页右上角配置面板临时填写，或 notebook 里调 set_llm_override
# 环境变量只作为无界面场景（评测脚本）的兜底，仓库不内置任何 key。

_PROVIDERS = {
    "qwen": {
        "base_url":          "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "default_model":     "qwen3.7-plus-2026-05-26",
        "api_key_env":       "DASHSCOPE_API_KEY",
        "supports_thinking": True,
    },
    "deepseek": {
        "base_url":          "https://api.deepseek.com/v1",
        "default_model":     "deepseek-chat",
        "api_key_env":       "DEEPSEEK_API_KEY",
        "supports_thinking": False,
    },
}

_DEFAULT_PROVIDER = "qwen"


def _build_llm(provider: str, model: str | None, api_key: str | None,
               enable_thinking: bool | None) -> ChatOpenAI:
    """按供应商拼一个 LLM 客户端，模型 / key 留空就用该供应商的默认值。"""
    spec = _PROVIDERS.get(provider, _PROVIDERS[_DEFAULT_PROVIDER])
    kwargs = {
        "model":       model or spec["default_model"],
        "base_url":    spec["base_url"],
        "api_key":     api_key or os.environ.get(spec["api_key_env"], ""),
        "temperature": 0.2,
    }
    # 只有认这个参数的供应商才带思考开关，DeepSeek 塞了会报错
    if spec["supports_thinking"]:
        kwargs["extra_body"] = {"enable_thinking": bool(enable_thinking)}
    return ChatOpenAI(**kwargs)


LLM = _build_llm(_DEFAULT_PROVIDER, None, None, None)

# 运行时覆盖（网页面板写入，无需重启）
_llm_override: dict               = {}
_llm_cache:    ChatOpenAI | None  = None


def get_llm() -> ChatOpenAI:
    """返回当前生效的 LLM 客户端。

    没人动过网页配置面板时，直接返回 settings.py 里写死的默认客户端；
    面板改过供应商、模型、API key 或思考模式之后，返回按这些改动重建的客户端，
    并缓存复用，避免每次调用都重新构造一个新客户端。

    Returns:
        当前生效的 ChatOpenAI 客户端。
    """
    if not _llm_override:
        return LLM
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = _build_llm(
            provider        = _llm_override.get("provider", _DEFAULT_PROVIDER),
            model           = _llm_override.get("model"),
            api_key         = _llm_override.get("api_key"),
            enable_thinking = _llm_override.get("enable_thinking"),
        )
    return _llm_cache


def set_llm_override(provider: str | None, model: str | None, api_key: str | None,
                     enable_thinking: bool | None) -> None:
    """把网页配置面板填的供应商、模型、API key、思考模式存成运行时覆盖。

    某一项留空（None 或空字符串）就保留默认值，不会被清掉。
    清空缓存是为了让 get_llm() 下次调用时按新的覆盖重新构造客户端，
    而不是继续返回改之前缓存的旧客户端。

    Args:
        provider: 要切换的供应商（qwen / deepseek），留空则继续用默认供应商。
        model: 要切换的模型名，留空则用该供应商的默认模型。
        api_key: 要使用的 API key，留空则继续用默认 key。
        enable_thinking: 是否开启思考模式，留空则继续用默认设置。
    """
    global _llm_override, _llm_cache
    _llm_override = {}
    if provider:                    _llm_override["provider"]        = provider
    if model:                       _llm_override["model"]           = model
    if api_key:                     _llm_override["api_key"]         = api_key
    if enable_thinking is not None: _llm_override["enable_thinking"] = enable_thinking
    _llm_cache = None


def current_config() -> dict:
    """返回当前生效的供应商 / 模型 / 思考模式，供网页面板回显。"""
    provider = _llm_override.get("provider", _DEFAULT_PROVIDER)
    spec     = _PROVIDERS.get(provider, _PROVIDERS[_DEFAULT_PROVIDER])
    return {
        "provider":          provider,
        "model":             _llm_override.get("model", spec["default_model"]),
        "enable_thinking":   bool(_llm_override.get("enable_thinking", False)),
        "supports_thinking": spec["supports_thinking"],
        "providers":         list(_PROVIDERS.keys()),
        "has_override":      bool(_llm_override),
    }
