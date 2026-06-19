"""RAG 统一入口。

本模块提供顶层函数 Agentic_RAG，对外屏蔽 vector / agentic 两种模式的实现差异：
    vector  — Run_Vector_Mode，纯向量检索，返回 chunk 列表并打印。
    agentic — Run_Agentic_Mode，多步 Agentic RAG，打印查询计划和最终答案。
"""

from rag.agent.modes.vector_mode import Run_Vector_Mode
from rag.agent.modes.agentic_mode import Run_Agentic_Mode
from rag.graph.nodes.route_task import Synthesize_Answer


def Agentic_RAG(query: str, mode: str = "agentic", history: list = [], top_k: int = 10) -> str:
    """统一 RAG 入口，根据 mode 分发到对应执行路径，返回最终答案字符串。

    Args:
        query: 用户问题。
        mode: "vector" 或 "agentic"。
        history: 多轮对话历史，含 role / content 字典列表，vector 模式忽略此参数。
    """

    if mode == "vector":
        chunks = Run_Vector_Mode(query, top_k=top_k)
        return _Print_Vector_Result(query, chunks)

    if mode == "agentic":
        return Run_Agentic_Mode(raw_query=query, history=history, top_k=top_k)

    raise ValueError(f"未知 mode: {mode!r}，只支持 'vector' 或 'agentic'")


def _Print_Vector_Result(query: str, chunks: list[dict]) -> str:
    """打印 vector mode 检索摘要，再调 LLM 综合回答，返回答案字符串。"""

    # 打印检索摘要，便于调试
    if chunks:
        print(f"[Vector] 检索到 {len(chunks)} 条结果\n")
        for i, chunk in enumerate(chunks, 1):
            meta = (
                f"[{i}] chunk_id={chunk.get('chunk_id', '?')}  "
                f"vol={chunk.get('volume', '?')}  "
                f"chunk={chunk.get('chunk_index', '?')}/{chunk.get('chunk_total', '?')}  "
                f"rerank={chunk.get('reranker_score', 0.0):.4f}  "
                f"rrf={chunk.get('rrf_score', 0.0):.6f}"
            )
            preview = chunk.get("content", "")[:100]
            print(meta)
            print(preview)
            print()
    else:
        print("[Vector] 未检索到相关内容\n")

    # LLM 综合回答，与 chunk 路由共用同一合成函数，答案走返回值不再打印
    answer = Synthesize_Answer(query, chunks)
    return answer