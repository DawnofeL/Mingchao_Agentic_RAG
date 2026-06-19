"""输出格式化。

本模块定义 Format_Chunks，把 Search_Chunks 返回的 list[dict] 格式化为可读文本，
适合直接在 notebook 中 print 或传给前端展示。

Agentic Mode 的流式事件转发后续在此文件中添加。
"""


def Format_Chunks(chunks: list[dict], show_content: bool = True) -> str:
    """把 chunk 列表格式化为可读文本。

    每条 chunk 输出一行元信息（卷号、chunk 位置、reranker 分、RRF 分），
    若 show_content 为 True，则在元信息下方附上 chunk 正文。

    Args:
        chunks:       Search_Chunks 返回的 list[dict]。
        show_content: 是否展示 chunk 正文，默认 True。
    Returns:
        格式化后的字符串，可直接 print 或在 Jupyter 中展示。
    """

    if not chunks:
        return "[chunk_rrf] 未召回相关内容。"

    lines = []

    for i, chunk in enumerate(chunks, 1):

        # 把元信息拼成一行，方便快速扫描排名和分数
        meta = (
            f"vol={chunk.get('volume', '?')}  "
            f"chunk={chunk.get('chunk_index', '?')}/{chunk.get('chunk_total', '?')}  "
            f"rerank={chunk.get('reranker_score', 0.0):.4f}  "
            f"rrf={chunk.get('rrf_score', 0.0):.6f}"
        )
        lines.append(f"[{i}] {meta}")

        if show_content:
            lines.append(chunk.get("content", ""))
            lines.append("")

    return "\n".join(lines)
