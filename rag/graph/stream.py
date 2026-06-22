"""输出格式化与流式转发。

本模块定义：
    Format_Chunks        — 把 Search_Chunks 返回的 list[dict] 格式化为可读文本。
    Stream_Agentic_Mode  — 流式运行 Agentic 顶层图，逐节点产出进度事件。
"""


def Format_Chunks(chunks: list[dict], show_content: bool = True) -> str:
    """把 chunk 列表格式化为可读文本。

    每条 chunk 输出一行元信息（卷号、chunk 位置、reranker 分、RRF 分），
    若 show_content 为 True，则在元信息下方附上 chunk 正文。纯展示用，
    方便在 notebook 里肉眼检查检索排序对不对，不参与任何业务逻辑。

    例：
        chunks = [{"volume": 2, "chunk_index": 5, "chunk_total": 12,
                   "reranker_score": 0.91, "rrf_score": 0.0163,
                   "content": "鄱阳湖之战中，朱元璋一方率先进军……"}]
                   
        Format_Chunks(chunks) 输出：
            [1] vol=2  chunk=5/12  rerank=0.9100  rrf=0.016300
            鄱阳湖之战中，朱元璋一方率先进军……

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


def Stream_Agentic_Mode(raw_query: str, history: list = [], top_k: int = 10):
    """流式运行 Agentic 顶层图，每个顶层节点完成时产出一个事件。

    用 agentic_graph.stream(stream_mode="updates")，QU 节点完成会先吐出 plan，
    随后对应分支节点完成会吐出 final_answer，前端可以先把查询计划展示出来，
    不用等整条检索链跑完。延迟 import agentic_graph，避免模块级循环引用。

    注意：编排和检索子图是在分支节点内部以函数方式调用的，所以它们内部的
    轮次 / 工具调用进度不会从这里逐条流出，那部分进度仍走各节点的 print 日志。

    Args:
        raw_query: 用户原始问题。
        history: 多轮对话历史。
        top_k: 检索返回数。
    Yields:
        dict，形如 {"node": 节点名, "update": 该节点返回的状态增量}。
        plan 在 qu 节点事件里，最终答案在分支节点事件的 final_answer 里。
    """

    from rag.graph.build import agentic_graph

    init: dict = {
        "raw_query":    raw_query,
        "history":      history,
        "top_k":        top_k,
        "plan":         {},
        "final_answer": "",
    }

    for update in agentic_graph.stream(init, stream_mode = "updates"):
        for node_name, node_output in update.items():
            yield {"node": node_name, "update": node_output}
