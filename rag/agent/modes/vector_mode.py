"""Vector Mode 入口。

本模块定义 Run_Vector_Mode，封装 vector_graph 的调用细节，
调用方只需传入问题，拿到 chunk 列表，不需要关心图状态结构。
"""

from rag.graph.build import vector_graph


def Run_Vector_Mode(query: str, top_k: int = 10) -> list[dict]:
    """运行 Vector Mode，返回 top-k chunk 列表。

    无历史记录、无 LLM 调用，纯代码检索路径。

    Args:
        query: 用户当前问题。
    Returns:
        list[dict]，每条含 chunk_id / volume / chapter / section /
        chunk_index / chunk_total / content / rrf_score / rrf_rank / reranker_score。
        无相关内容时返回空列表。
    """

    result = vector_graph.invoke({"raw_query": query, "chunks": [], "top_k": top_k})
    return result["chunks"]
