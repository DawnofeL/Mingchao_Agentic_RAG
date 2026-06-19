"""Agentic Mode 入口。

本模块定义 Run_Agentic_Mode，封装 Agentic RAG 的完整调用路径：
    Query_Understanding_Node → Route_Task
调用方只需传入问题和历史，即可触发完整的理解→路由→检索流程。
"""

import json

from rag.graph.nodes.query_understanding import Query_Understanding_Node
from rag.graph.nodes.route_task import Route_Task


def Run_Agentic_Mode(raw_query: str, history: list = [], top_k: int = 10) -> str:
    """运行 Agentic Mode：QU 解析 → 任务路由 → 检索执行，返回最终答案字符串。

    Args:
        raw_query: 用户原始问题。
        history: 多轮对话历史，每条含 role / content，为空时跳过指代消解。
    """

    plan = Query_Understanding_Node(raw_query, history)

    print("=== Query Plan ===")
    print(json.dumps(plan, indent = 2, ensure_ascii = False))
    print()

    return Route_Task(plan, history, top_k=top_k)