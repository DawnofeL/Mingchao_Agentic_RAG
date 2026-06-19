"""LangGraph 状态定义。

本模块定义各模式图节点共享的状态 TypedDict。
Agentic Mode 的状态后续在此文件中添加。
"""

from typing import TypedDict


class VectorState(TypedDict):
    raw_query: str   # 用户输入问题
    chunks:    list  # Search_Chunks 返回的 chunk 列表，图执行结束后读取此字段
    top_k:     int   # 最终返回 chunk 数，传入 Search_Chunks
