"""Agentic Mode 入口。

本模块定义 Run_Agentic_Mode，调用 build.py 里编译好的 agentic_graph，
顶层图内部完成 理解→路由→检索/编排 的完整流程，调用方只需传入问题和历史。
"""

from rag.graph.build import agentic_graph


def Run_Agentic_Mode(raw_query: str, history: list = [], top_k: int = 10) -> str:
    """运行 Agentic Mode：调用顶层图，返回最终答案字符串。

    Args:
        raw_query: 用户原始问题。
        history: 多轮对话历史，每条含 role / content，为空时跳过指代消解。
        top_k: 检索返回数，透传给各检索分支。
    """

    init: dict = {
        "raw_query":    raw_query,
        "history":      history,
        "top_k":        top_k,
        "plan":         {},
        "final_answer": "",
    }
    final_state = agentic_graph.invoke(init)
    return final_state["final_answer"]