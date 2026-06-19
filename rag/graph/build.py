"""LangGraph 图构建器。

本模块定义 Vector Mode 的检索图，模块加载时编译为全局实例 vector_graph。

本模块定义以下函数：

    _Retrieve_Node       检索节点，调用 Search_Chunks 并把结果写入 chunks 字段。
    Build_Vector_Graph   组装 LangGraph StateGraph，编译并返回可执行图对象。

图结构（Vector Mode）：
    START → _Retrieve_Node → END

Agentic Mode 图后续在此文件中添加。
"""

from langgraph.graph import END, START, StateGraph

from rag.graph.state import VectorState
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
