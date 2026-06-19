"""FastAPI 应用：路由 + SSE 推送。

端点：
    GET  /                  → web/index.html
    GET  /health            → {"status":"ok", "ready": bool}
    GET  /startup-progress  → SSE 流，推送模型加载进度
    POST /chat              → SSE 流，实时推送 RAG 运行日志和最终答案
"""

import asyncio
import json
import queue
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

import rag.config.settings as _settings
from rag.agent.rag import Agentic_RAG
from server.schemas import ChatRequest

_WEB = Path(__file__).resolve().parents[1] / "web"

# 启动状态：由 app.py 的加载线程写入，SSE 端点读取
_startup_q:    queue.Queue = queue.Queue()
_startup_done: bool        = False

app = FastAPI(title="Ming RAG")

# web/ 目录挂载为静态文件（CSS / JS）
app.mount("/static", StaticFiles(directory=_WEB), name="static")


# ── stdout 捕获器 ─────────────────────────────────────────────────────────────

class _SSECapture:
    """替换 sys.stdout，将 print() 输出实时投入 SSE 队列，同时保留终端回显。"""

    def __init__(self, q: queue.Queue, real) -> None:
        self._q    = q
        self._real = real

    def write(self, text: str) -> None:
        self._real.write(text)          # 终端仍然可见
        if text and text.strip():
            self._q.put(text)

    def flush(self) -> None:
        self._real.flush()


# ── 路由 ──────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": _startup_done}


@app.get("/startup-progress")
async def startup_progress() -> EventSourceResponse:
    """SSE：推送模型加载进度，直到 done=True 为止。"""

    async def _gen():
        while True:
            try:
                msg = _startup_q.get_nowait()
                yield {"data": json.dumps(msg, ensure_ascii=False)}
                if msg.get("done"):
                    break
            except queue.Empty:
                await asyncio.sleep(0.12)

    return EventSourceResponse(_gen(), sep="\n")


@app.get("/llm-config")
def get_llm_config() -> dict:
    """返回当前生效的 LLM 配置（base + override 合并结果）。"""
    base = _settings._LLM_BASE_KWARGS
    ov   = _settings._llm_override
    return {
        "model":           ov.get("model",           base["model"]),
        "enable_thinking": ov.get("enable_thinking", base["extra_body"]["enable_thinking"]),
        "has_override":    bool(ov),
    }


@app.post("/llm-config")
def set_llm_config(req: dict) -> dict:
    """运行时更新 LLM 配置，无需重启。空字符串 = 恢复默认。"""
    _settings.set_llm_override(
        model           = req.get("model")    or None,
        api_key         = req.get("api_key")  or None,
        enable_thinking = req.get("enable_thinking"),
    )
    return {"ok": True}


@app.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """SSE：运行 RAG，把 print() 输出实时推到前端；最后发 event:done。

    前端按 req.cot 决定展示哪些内容：
        cot=True  → 快速打出日志，再慢速打出最终答案
        cot=False → 缓冲所有日志，只慢速打出最终答案
    """

    if not _startup_done:
        raise HTTPException(503, detail="模型尚未加载完成，请稍候")

    q: queue.Queue = queue.Queue()
    history = [{"role": m.role, "content": m.content} for m in req.history]

    def _run() -> None:
        real       = sys.stdout
        sys.stdout = _SSECapture(q, real)
        answer = ""
        try:
            answer = Agentic_RAG(query=req.query, mode=req.mode, history=history)
        except Exception as e:
            # 崩溃时把错误当答案走独立通道，日志区只留一行排查信息
            err = f"（生成失败：{type(e).__name__}: {e}）"
            print(f"[Error] {err}")
            answer = err
        finally:
            sys.stdout = real
            q.put(("answer", answer or ""))   # 先推最终答案，走独立通道
            q.put(None)                        # 哨兵：通知 SSE 生成器停止

    threading.Thread(target=_run, daemon=True).start()

    async def _gen():
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            if item is None:
                yield {"event": "done", "data": ""}
                break
            if isinstance(item, tuple) and item[0] == "answer":
                yield {"event": "answer", "data": json.dumps({"text": item[1]}, ensure_ascii=False)}
                continue
            yield {"event": "log", "data": json.dumps({"text": item}, ensure_ascii=False)}

    return EventSourceResponse(_gen(), sep="\n")
