"""LangGraph 图构建器。

本模块定义两张图，模块加载时分别编译为全局实例 vector_graph 和 agentic_graph。

Vector Mode 图结构：
    START → _Retrieve_Node → END

Agentic Mode 图结构（顶层）：
    START → qu → 条件路由 → {single_people / single_timeline / single_direct / orchestrate} → END
其中编排分支和各检索分支内部都调用对应子图（orchestrator / people / timeline），
顶层图只负责"理解问题 → 决定走哪条线"。
"""

import json

from langgraph.graph import END, START, StateGraph

from rag.graph.nodes.orchestrator import Run_Orchestrator
from rag.graph.nodes.query_understanding import Query_Understanding_Node
from rag.graph.nodes.route_task import _Run_Direct, _Run_People, _Run_Timeline
from rag.graph.state import AgenticState, VectorState
from rag.retrieval.chunk_rrf import Search_Chunks


# ── 节点函数 ──────────────────────────────────────────────────────────────────

def _Retrieve_Node(state: VectorState) -> dict:
    """检索节点，调用 Search_Chunks 并把结果写入 chunks 字段。

    Args:
        state: 当前图状态，读取 raw_query 字段。
    Returns:
        {"chunks": list[dict]}，LangGraph 会把这个 dict merge 回 state。
    """

    chunks = Search_Chunks(state["raw_query"], top_k=state["top_k"])
    return {"chunks": chunks}


# ── 图构建 ────────────────────────────────────────────────────────────────────

def Build_Vector_Graph():
    """组装 Vector Mode 的 LangGraph StateGraph 并编译。

    StateGraph 是 LangGraph 的状态图，每个节点接收 VectorState 并返回需要更新的字段。
    Vector Mode 只有一个节点，图结构是 START → retrieve → END。

    Returns:
        LangGraph CompiledGraph 对象。
    """

    builder = StateGraph(VectorState)

    builder.add_node("retrieve", _Retrieve_Node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", END)

    return builder.compile()


# 模块加载时编译图，后续直接用 vector_graph.invoke() 调用
vector_graph = Build_Vector_Graph()


# ── Agentic Mode 节点 ─────────────────────────────────────────────────────────

def _QU_Node(state: AgenticState) -> dict:
    """意图识别节点：把原始问题解析成结构化查询计划。"""

    plan = Query_Understanding_Node(state["raw_query"], state["history"])

    print("=== Query Plan ===")
    print(json.dumps(plan, indent = 2, ensure_ascii = False))
    print()

    return {"plan": plan}


def _Single_People_Node(state: AgenticState) -> dict:
    """单任务 · people 分支。"""

    task = state["plan"]["tasks"][0]
    print("[Router] 单任务 · intention=people")
    print()
    answer = _Run_People(task["task"], query_kind = task.get("query_kind", "fact"), top_k = state["top_k"])
    return {"final_answer": answer}


def _Single_Timeline_Node(state: AgenticState) -> dict:
    """单任务 · timeline 分支。"""

    task = state["plan"]["tasks"][0]
    print("[Router] 单任务 · intention=timeline")
    print()
    answer = _Run_Timeline(task["task"], query_kind = task.get("query_kind", "fact"), top_k = state["top_k"])
    return {"final_answer": answer}


def _Single_Direct_Node(state: AgenticState) -> dict:
    """单任务 · direct 分支，基于历史直接回复，无检索。"""

    task = state["plan"]["tasks"][0]
    print("[Router] 单任务 · intention=direct")
    print()
    answer = _Run_Direct(task["task"], state["history"])
    return {"final_answer": answer}


def _Orchestrate_Node(state: AgenticState) -> dict:
    """多任务分支：转交编排子图做拓扑调度。"""

    print("[Router] 多任务 → 路由至 Orchestrator")
    answer = Run_Orchestrator(state["plan"], state["history"], top_k = state["top_k"])
    return {"final_answer": answer}


def _Route_After_QU(state: AgenticState) -> str:
    """按 task_type / intention 决定走编排分支还是某个单任务分支。"""

    plan = state["plan"]

    if plan["task_type"] == "subtasks":
        return "orchestrate"

    intention = plan["tasks"][0]["intention"]
    branch = {
        "people":   "single_people",
        "timeline": "single_timeline",
        "direct":   "single_direct",
    }.get(intention)

    if branch is None:
        raise ValueError(f"未知 intention: {intention!r}")

    return branch


def Build_Agentic_Graph():
    """组装 Agentic Mode 顶层图并编译。

    START → qu → 条件路由 → 四个分支之一 → END。
    分支节点内部各自调用编排 / 检索子图，顶层图只做理解与路由。

    Returns:
        LangGraph CompiledGraph 对象。
    """

    builder = StateGraph(AgenticState)

    builder.add_node("qu",              _QU_Node)
    builder.add_node("single_people",   _Single_People_Node)
    builder.add_node("single_timeline", _Single_Timeline_Node)
    builder.add_node("single_direct",   _Single_Direct_Node)
    builder.add_node("orchestrate",     _Orchestrate_Node)

    builder.add_edge(START, "qu")
    builder.add_conditional_edges(
        "qu", _Route_After_QU,
        ["single_people", "single_timeline", "single_direct", "orchestrate"],
    )
    for branch in ("single_people", "single_timeline", "single_direct", "orchestrate"):
        builder.add_edge(branch, END)

    return builder.compile()


# 模块加载时编译图，后续直接用 agentic_graph.invoke() 调用
agentic_graph = Build_Agentic_Graph()
