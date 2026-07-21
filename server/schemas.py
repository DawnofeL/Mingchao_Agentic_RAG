from pydantic import BaseModel


class HistoryMessage(BaseModel):
    role:    str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    query:   str
    mode:    str  = "agentic"   # "vector" | "agentic"
    history: list[HistoryMessage] = []
    cot:     bool = True        # 前端是否展示 agent 日志
    conversation_id: int | None = None   # 归属的历史对话编号，留空则后端新开一场
