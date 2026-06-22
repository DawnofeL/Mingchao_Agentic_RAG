"""把项目里所有 LangGraph 图手工拼成一张合成可视化图。

这里不是调用 LangGraph 原生的子图嵌套 API（项目本身也没用那套机制，
顶层图调用子图是靠普通函数调用 + state 手动转换，见 route_people.py /
route_task.py），纯粹是为了看图方便，手写 mermaid 文本，把每张子图的节点
用 subgraph 语法圈起来上底色，再用虚线箭头表示"顶层节点调用了这张子图"。
跟 render_graphs.py 不一样，这里不读任何编译好的图对象，节点和边都是按
各文件里 add_node / add_edge 抄下来的，图结构改了要手动同步这份文件。

用法：
    python output/graph_viz/render_combined_graph.py

渲染走本地 mermaid-cli（npx @mermaid-js/mermaid-cli），不走 mermaid.ink 的
在线接口。试过在线接口，复杂图加中文注释偶发 404，本地渲染更稳定。
"""

import subprocess
from pathlib import Path

_MERMAID = """
flowchart TD

    %% ===== 顶层图：agentic_graph（build.py） =====
    top_start([START]) --> qu["qu 意图识别"]
    qu --> top_route{"条件路由"}
    top_route --> single_people["single_people"]
    top_route --> single_timeline["single_timeline"]
    top_route --> single_direct["single_direct"]
    top_route --> orchestrate["orchestrate"]
    single_people --> top_end([END])
    single_timeline --> top_end
    single_direct --> top_end
    orchestrate --> top_end

    %% ===== Orchestrator 子图（orchestrator.py） =====
    subgraph SG_ORCH [" Orchestrator 子图 "]
        o_start([START]) --> orchestrator["orchestrator"]
        orchestrator -->|worker| worker["worker"]
        orchestrator -->|synthesize| synthesize["synthesize"]
        worker --> orchestrator
        synthesize --> o_end([END])
    end

    %% ===== People 子图（route_people.py） =====
    subgraph SG_PEOPLE [" People 子图 "]
        p_start([START]) --> p_first["first_call"]
        p_first -->|execute| p_exec["execute_tool"]
        p_first -->|end| p_end1([END])
        p_exec --> p_judge["judge"]
        p_judge -->|retry| p_retry["execute_retry"]
        p_judge -->|partial| p_partial["make_partial"]
        p_judge -->|end| p_end2([END])
        p_retry --> p_final["final_judge"]
        p_final -->|partial| p_partial
        p_final -->|end| p_end3([END])
        p_partial --> p_end4([END])
    end

    %% ===== Timeline 子图（route_timeline.py） =====
    subgraph SG_TIMELINE [" Timeline 子图 "]
        t_start([START]) --> t_first["first_call"]
        t_first -->|execute| t_exec["execute_tool"]
        t_first -->|end| t_end1([END])
        t_exec --> t_judge["judge"]
        t_judge -->|partial| t_partial["make_partial"]
        t_judge -->|end| t_end2([END])
        t_partial --> t_end3([END])
    end

    %% ===== Vector 图（build.py，独立入口，不挂在顶层图下） =====
    subgraph SG_VECTOR [" Vector 图（独立入口） "]
        v_start([START]) --> v_retrieve["retrieve"] --> v_end([END])
    end

    %% ===== 顶层节点 → 子图，虚线表示函数调用而非 LangGraph 原生边 =====
    single_people  -. 调用 .-> p_start
    single_timeline -. 调用 .-> t_start
    orchestrate    -. 调用 .-> o_start

    %% ===== 节点上色 =====
    classDef topStyle      fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef orchStyle     fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef peopleStyle   fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef timelineStyle fill:#fce7f3,stroke:#be185d,color:#831843
    classDef vectorStyle   fill:#ede9fe,stroke:#6d28d9,color:#4c1d95

    class qu,single_people,single_timeline,single_direct,orchestrate topStyle
    class orchestrator,worker,synthesize orchStyle
    class p_first,p_exec,p_judge,p_retry,p_final,p_partial peopleStyle
    class t_first,t_exec,t_judge,t_partial timelineStyle
    class v_retrieve vectorStyle

    %% ===== 子图整体底色，配合 classDef 让框本身也有颜色 =====
    style SG_ORCH     fill:#fffbeb,stroke:#b45309
    style SG_PEOPLE   fill:#f0fdf4,stroke:#15803d
    style SG_TIMELINE fill:#fdf2f8,stroke:#be185d
    style SG_VECTOR   fill:#f5f3ff,stroke:#6d28d9
"""

if __name__ == "__main__":
    out_dir  = Path(__file__).resolve().parent
    mmd_path = out_dir / "0_combined.mmd"
    out_path = out_dir / "0_combined.png"

    mmd_path.write_text(_MERMAID, encoding = "utf-8")

    # 本地跑 mermaid-cli，等价于 npx -y @mermaid-js/mermaid-cli -i xx.mmd -o xx.png
    subprocess.run(
        ["npx", "-y", "@mermaid-js/mermaid-cli",
         "-i", str(mmd_path), "-o", str(out_path), "-b", "white"],
        check = True,
    )
    print(f"已生成 {out_path.name}（{out_path.stat().st_size} 字节）")
