"""把项目里编译好的 LangGraph 图渲染成 PNG。

子图是以函数调用方式嵌套的，顶层图里 orchestrate / single_* 只是单个节点，
内部要单独看，所以这里把五张图各渲染一张：顶层、编排、人物、时间线、向量。
改了图结构后重跑本脚本即可刷新图片。

用法（在 SEDNA 环境下）：
    python output/graph_viz/render_graphs.py
"""

import sys
from pathlib import Path

# 把仓库根目录加进 sys.path，保证 from rag.xxx import 能找到
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from rag.graph.build import agentic_graph, vector_graph
import rag.graph.nodes.orchestrator as orch
import rag.graph.nodes.route_people as rp
import rag.graph.nodes.route_timeline as rt


# 图名 → 编译好的图对象
graphs = {
    "1_agentic_top": agentic_graph,
    "2_orchestrator": orch._orchestrator_graph,
    "3_people": rp._people_graph,
    "4_timeline": rt._timeline_graph,
    "5_vector": vector_graph,
}

out_dir = Path(__file__).resolve().parent

for name, graph in graphs.items():
    png = graph.get_graph().draw_mermaid_png()
    path = out_dir / f"{name}.png"
    path.write_bytes(png)
    print(f"已生成 {path.name}（{len(png)} 字节）")
