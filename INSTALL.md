# INSTALL · 给 Coding Agent 的安装指南

严格按顺序执行，遇到不确定的地方先检测再决定，不要臆测。全程不得删除或覆盖用户已有的任何 conda 环境，本项目一律新建独立环境 `mingchao_rag`。



---

## 0. 前置检测

先探测运行环境，后续步骤据此分支：

1. 操作系统：Windows / WSL / Linux / macOS。
   - **Windows 原生不支持 Milvus Lite**，必须在 WSL 内运行。若检测到原生 Windows，提示用户改用 WSL。
2. 硬件加速：
   - 有 NVIDIA 显卡（`nvidia-smi` 可用）→ 走 CUDA。
   - Apple Silicon（macOS arm64，`uname -m` 为 `arm64`）→ 走 MPS。
   - 其余（Intel Mac / 无独显 / 前面失败）→ 走 CPU。
3. 是否已安装 conda（优先）或仅有 venv。

---

## 1. 克隆仓库

先询问用户：「你想把项目放到哪个文件夹目录？如果你已经下载，请告诉我本地的项目文件地址」，**必须**等用户给出路径后再往下执行，**严禁擅自推进**。

当用户提供地址后，执行：

```bash
git clone https://github.com/DawnofeL/Mingchao_Agentic_RAG.git <用户指定路径>/Mingchao_Agentic_RAG
cd <用户指定路径>/Mingchao_Agentic_RAG
```

若用户已通过 Download ZIP 解压，跳过 clone。

---

## 2. 新建独立 Python 环境

conda（优先）：

```bash
conda create -n mingchao_rag python=3.10 -y
conda activate mingchao_rag
```

没有 conda 时用 venv：

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
```

---

## 3. 安装 torch（按硬件选其一）

torch 不在 `requirements.txt` 里，按检测结果单独装：

```bash
# CUDA（有 NVIDIA 显卡）
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Apple Silicon（macOS arm64）
pip install torch

# CPU（无 GPU，或上面失败时的兜底）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

若 CUDA / MPS 安装失败，退到 CPU 版本。代码侧会自动按 `cuda → mps → cpu` 选择设备。

---

## 4. 安装其余依赖

```bash
pip install -r requirements.txt
```

---

## 5. 下载模型

两个 BGE 模型共约 6.5GB：

```bash
python -m pip install -U huggingface_hub

hf download BAAI/bge-m3 --local-dir model/BAAI_bge-m3
hf download BAAI/bge-reranker-v2-m3 --local-dir model/BAAI_bge-reranker-v2-m3
```

---

## 6. API key

安装阶段无需配置，不经手、不索取。启动后提示用户在网页右上角配置面板填写。

---

## 7. 启动

直接运行：

```bash
cd <用户指定路径>/Mingchao_Agentic_RAG
conda activate mingchao_rag && python app.py
```

venv 用户替换为对应激活命令。启动后自动打开 `http://localhost:8000`，首次启动耗时几秒建本地库。

---

## 8. 完成后向用户报告

- 项目路径、环境名（`mingchao_rag`）、操作系统、torch 设备（CUDA / MPS / CPU）。
- Milvus 默认走 Lit，；需要 Docker 的话在 `rag/config/settings.py` 把 `MILVUS_MODE` 改为 `"docker"`。
- API key 和模型选择在网页右上角配置面板填写，请使用阿里通义千问 Qwen 系列模型。
- 如需卸载，告知用户直接说「卸载」，agent 运行以下命令彻底清除本项目新建的一切，不影响本机其他任何东西，：
  ```bash
  conda deactivate
  conda env remove -n mingchao_rag -y
  rm -rf <用户指定路径>/Mingchao_Agentic_RAG
  rm -rf ~/.cache/huggingface
  ```

​	运行完后自检一下是否清理干净，然后向用户汇报，最后加一句 “感谢您花时间体验本项目”。