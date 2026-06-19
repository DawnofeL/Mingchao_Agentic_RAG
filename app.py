"""一键启动入口。

    python app.py

流程：
    1. 预导入 server.app（确保模块在主线程初始化，无竞争）
    2. 后台线程逐步加载 BGE-M3 / Reranker / Milvus，进度实时写入 _startup_q
    3. uvicorn 立刻启动，浏览器打开 loading 页面
    4. 前端通过 GET /startup-progress SSE 接收进度并渲染进度条
    5. 模型全部就绪后 _startup_done 置 True，前端自动切换到主界面
"""

import platform
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import uvicorn

import server.app as _srv              # 预导入，确保模块单例在主线程创建
from rag.config.settings import BGE_MODEL_PATH, RERANKER_MODEL_PATH
from rag.retrieval.chunk_rrf import _Get_Collection, Inject_Models


def _load_models() -> None:
    """后台加载所有资源，进度通过 _startup_q 推送给前端 SSE。"""

    try:
        from FlagEmbedding import BGEM3FlagModel, FlagReranker

        _srv._startup_q.put({"step": "加载 BGE-M3 编码器", "progress": 10})
        device  = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = BGEM3FlagModel(BGE_MODEL_PATH, use_fp16=(device == "cuda"))

        _srv._startup_q.put({"step": "加载 Reranker 精排模型", "progress": 50})
        reranker = FlagReranker(RERANKER_MODEL_PATH, use_fp16=(device == "cuda"))

        _srv._startup_q.put({"step": "初始化 Milvus 向量库", "progress": 80})
        collection = _Get_Collection()

        _srv._startup_q.put({"step": "注入模型资源", "progress": 95})
        Inject_Models(encoder=encoder, reranker=reranker, collection=collection)

        _srv._startup_done = True
        _srv._startup_q.put({"step": "就绪", "progress": 100, "done": True})

    except Exception as e:
        _srv._startup_q.put({"error": str(e), "done": True})
        raise


def _open_browser():
    url = "http://localhost:8000"
    # WSL 环境用 Windows cmd 打开浏览器
    if "microsoft" in platform.uname().release.lower():
        subprocess.Popen(["cmd.exe", "/c", "start", url])
    else:
        import webbrowser
        webbrowser.open(url)


if __name__ == "__main__":
    threading.Thread(target=_load_models, daemon=True).start()
    threading.Timer(1.5, _open_browser).start()
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")
