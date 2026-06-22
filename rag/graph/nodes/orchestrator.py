"""Orchestrator 编排子图。

本模块把多子任务的拓扑调度实现成一张有环的 LangGraph 子图，只在 task_type == "subtasks"
时介入。核心是 orchestrator-worker 循环：orchestrator 算出本轮 ready 任务并用 Send 扇出，
worker 并行执行后把结果合并回 pool，再回到 orchestrator 算下一轮，直到没有待办任务。

节点：
    orchestrator — 算 ready 任务、解引用 / 枚举增生、写阻塞结果、产出本轮 jobs。
    worker       — 执行单个 job（一条子任务），结果经 reducer 合并进 pool。
    synthesize   — 读结果池，调 LLM 合成最终回答。

复用的纯函数（逻辑不变，节点直接调用）：
    _Resolve_References      — LLM 调用：指代还原 / 枚举增生 / 阻塞判定
    _Execute_Task            — 按 intention 派发到对应 _Run_* 执行器
    _Synthesize_Final_Answer — LLM 调用：读结果池合成最终回答

入口 Run_Orchestrator 保持原有签名与返回值不变，上游 route_task 无需改动。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from rag.config.settings import get_llm
from rag.graph.citation_check import Extract_Cited_Ids, Validate_Citations
from rag.graph.state import OrchestratorState


_RESOLVE_SKILL_PATH = Path(__file__).resolve().parents[2] / "agent" / "skills" / "resolve_references.md"
_FINAL_SKILL_PATH   = Path(__file__).resolve().parents[2] / "agent" / "skills" / "final_answer.md"


@dataclass
class TaskResult:
    task_id:   str
    task:      str
    intention: str
    answer:    str
    blocked:   bool = False


# ── LangGraph 编排子图：orchestrator-worker 循环 + Send 扇出 ───────────────────

def _Node_Orchestrator(state: OrchestratorState) -> dict:
    """算出本轮 ready 任务，解引用 / 枚举增生 / 写阻塞结果，产出本轮 jobs。

    与原 while 循环单轮逻辑一一对应。route 字段告诉后面的路由怎么走：
        "dispatch"   → 有 job，扇出给 worker 并行执行
        "loop"       → 本轮全阻塞但还有 pending，回到 orchestrator 算下一轮
        "synthesize" → 死锁或全部完成，进入终答合成
    """

    pending   = state["pending"]
    pool      = state["pool"]
    refined_q = state["refined_query"]

    # 没有待办任务，直接进合成（对应原 while pending 自然退出，不算死锁）
    if not pending:
        return {"jobs": [], "route": "synthesize"}

    round_num = state["round_num"] + 1

    ready = [t for t in pending if all(dep in pool for dep in t["depends_on"])]

    # 死锁：pending 非空却一个能跑的都挑不出来，直接进合成（与原 break 行为一致）
    if not ready:
        print(f"\n[Orchestrator] 调度死锁，剩余：{[t['task_id'] for t in pending]}")
        return {"round_num": round_num, "jobs": [], "route": "synthesize"}

    ready_ids   = {t["task_id"] for t in ready}
    new_pending = [t for t in pending if t["task_id"] not in ready_ids]

    print(f"\n[Orchestrator] 第 {round_num} 轮  ready = {[t['task_id'] for t in ready]}")

    jobs:         list[dict] = []
    blocked_pool: dict       = {}

    for t in ready:
        if not t["depends_on"]:
            jobs.append({"parent": t, "text": t["task"]})
            continue

        resolved = _Resolve_References(t, pool, refined_q)

        if not resolved:
            blocked_pool[t["task_id"]] = [TaskResult(
                task_id   = t["task_id"],
                task      = t["task"],
                intention = t["intention"],
                answer    = "根据现有资料，无法回答此部分。",
                blocked   = True,
            )]
            print(f"[Orchestrator] {t['task_id']} 已阻塞（上游结果不足）")
            continue

        if len(resolved) == 1:
            print(f"[Orchestrator] {t['task_id']} 指代还原 → {resolved[0]}")
        else:
            print(f"[Orchestrator] {t['task_id']} 枚举增生 → {len(resolved)} 条")
            for i, text in enumerate(resolved, 1):
                print(f"  [{i}] {text}")

        for text in resolved:
            jobs.append({"parent": t, "text": text})

    updates: dict = {"round_num": round_num, "pending": new_pending, "jobs": jobs}
    if blocked_pool:
        updates["pool"] = blocked_pool   # 经 merge_pool reducer 合并

    if jobs:
        print(f"\n[Orchestrator] 并发执行 {len(jobs)} 个 job")
        updates["route"] = "dispatch"
    elif new_pending:
        # 本轮 ready 全阻塞，没 job 可派，但还有依赖它们的任务，回去算下一轮
        updates["route"] = "loop"
    else:
        updates["route"] = "synthesize"

    return updates


def _Node_Worker(payload: dict) -> dict:
    """执行单个 job（一条已解引用的子任务），结果经 reducer 合并进 pool。

    payload 由路由从 jobs 构造，含 parent / resolved_text / history / top_k。
    """

    result = _Execute_Task(
        payload["parent"],
        payload["resolved_text"],
        payload["history"],
        payload["top_k"],
    )
    return {"pool": {result.task_id: [result]}}


def _Node_Synthesize(state: OrchestratorState) -> dict:
    """读结果池，调 LLM 合成最终回答。"""

    print(f"\n[Orchestrator] 所有子任务完成，进入终答合成\n")
    answer = _Synthesize_Final_Answer(state["refined_query"], state["pool"])
    return {"final_answer": answer}


def _Route_From_Orchestrator(state: OrchestratorState):
    """按 orchestrator 给的 route 扇出 worker、回环、或进入合成。"""

    route = state["route"]

    if route == "dispatch":
        return [
            Send("worker", {
                "parent":        job["parent"],
                "resolved_text": job["text"],
                "history":       state["history"],
                "top_k":         state["top_k"],
            })
            for job in state["jobs"]
        ]

    if route == "loop":
        return "orchestrator"

    return "synthesize"


def _Build_Orchestrator_Graph():
    """组装并编译多子任务编排子图（有环：worker 跑完回到 orchestrator）。"""

    builder = StateGraph(OrchestratorState)

    builder.add_node("orchestrator", _Node_Orchestrator)
    builder.add_node("worker",       _Node_Worker)
    builder.add_node("synthesize",   _Node_Synthesize)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator", _Route_From_Orchestrator,
        ["worker", "orchestrator", "synthesize"],
    )
    builder.add_edge("worker", "orchestrator")
    builder.add_edge("synthesize", END)

    return builder.compile()


# 模块加载时编译子图，Run_Orchestrator 直接调用
_orchestrator_graph = _Build_Orchestrator_Graph()


def Run_Orchestrator(plan: dict, history: list = [], top_k: int = 10) -> str:
    """多子任务拓扑调度主入口，保持原有签名与返回值不变。

    构造初始状态后调用编排子图，返回终答合成结果。recursion_limit 调高到 100，
    因为每一轮编排是 orchestrator + worker 两个 superstep，默认 25 会限制轮数。

    Args:
        plan:    QU 输出的查询计划，含 refined_query / task_type / tasks。
        history: 多轮对话历史，供 direct intention 子任务使用。
    Returns:
        终答合成后的答案字符串。
    """

    init: OrchestratorState = {
        "refined_query": plan["refined_query"],
        "history":       history,
        "top_k":         top_k,
        "pending":       list(plan["tasks"]),
        "pool":          {},
        "jobs":          [],
        "round_num":     0,
        "route":         "",
        "final_answer":  "",
    }
    final_state = _orchestrator_graph.invoke(init, {"recursion_limit": 100})
    return final_state["final_answer"]


def _Resolve_References(
    downstream: dict,
    pool: dict[str, list[TaskResult]],
    refined_query: str,
) -> list[str]:
    """LLM 解引用：指代还原 / 枚举增生 / 阻塞判定。

    Args:
        downstream:    下游 task_item 字典，含 task / query_kind / depends_on。
        pool:          当前结果池，含所有已完成的上游答案。
        refined_query: QU 输出的精化问题，提供全局语境。
    Returns:
        []        → 上游不足，下游阻塞
        [text]    → 指代还原，1 条可执行任务
        [t1, ...] → 枚举增生，N 条可执行任务
    """

    upstream_parts = []
    for dep_id in downstream["depends_on"]:
        results = pool.get(dep_id, [])
        answers = "\n".join(r.answer for r in results)
        upstream_parts.append(f"上游任务 {dep_id}：\n{answers}")

    user_content = (
        f"用户原始问题：{refined_query}\n\n"
        + "\n\n".join(upstream_parts)
        + f"\n\n下游待执行任务：{downstream['task']}\n"
        + f"下游任务性质：{downstream['query_kind']}"
    )

    system_prompt = _RESOLVE_SKILL_PATH.read_text(encoding="utf-8")

    print(f"\n[Resolve] {downstream['task_id']} ← deps={downstream['depends_on']}")

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])

    raw = response.content.strip() if response.content else ""
    resolved = _Parse_Resolved(raw)
    print(f"[Resolve] 输出：{resolved}")
    return resolved


def _Parse_Resolved(raw: str) -> list[str]:
    """从 LLM 输出中提取 resolved_tasks 列表，容忍格式不规范。"""

    try:
        data = json.loads(raw)
        return [s for s in data.get("resolved_tasks", []) if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, AttributeError):
        pass

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return [s for s in data.get("resolved_tasks", []) if isinstance(s, str) and s.strip()]
        except (json.JSONDecodeError, AttributeError):
            pass

    return []


def _Execute_Task(parent: dict, resolved_text: str, history: list, top_k: int = 10) -> TaskResult:
    """按 intention 派发到对应执行器，返回 TaskResult。

    延迟导入 route_task 的 _Run_* 函数，避免与 route_task.py 形成模块级循环引用。
    """

    from rag.graph.nodes.route_task import _Run_Direct, _Run_People, _Run_Timeline

    intention  = parent["intention"]
    task_id    = parent["task_id"]
    query_kind = parent.get("query_kind", "fact")
    preview    = resolved_text[:60] + ("…" if len(resolved_text) > 60 else "")
    print(f"\n[Task] {task_id} ({intention}) → {preview}")

    if intention == "people":
        answer = _Run_People(resolved_text, query_kind=query_kind, top_k=top_k)
    elif intention == "timeline":
        answer = _Run_Timeline(resolved_text, query_kind=query_kind, top_k=top_k)
    elif intention == "direct":
        answer = _Run_Direct(resolved_text, history)
    else:
        answer = f"（未知 intention：{intention}）"

    return TaskResult(
        task_id   = task_id,
        task      = resolved_text,
        intention = intention,
        answer    = answer or "根据现有资料，无法回答此部分。",
    )


def _Synthesize_Final_Answer(refined_query: str, pool: dict[str, list[TaskResult]]) -> str:
    """读结果池，调 LLM 合成最终回答并返回，答案不打印到日志。

    合成规则只允许原样复述子任务答案里已有的引用锚点，不允许新增。
    生成后逐字段校验最终答案的 [people_id=N] / [event_id=N] / [chunk_id=N]
    是否全部来自结果池里子任务答案本身已出现过的 id，校验不过打日志报警并拒答。
    """

    parts:   list[str] = []
    blocked: list[str] = []

    for task_id, results in sorted(pool.items()):
        for r in results:
            if r.blocked:
                blocked.append(f"- {task_id}：\"{r.task}\"")
            else:
                parts.append(f"[{task_id}] 任务：\"{r.task}\"\n     回答：{r.answer}")

    evidence_text = "\n\n".join(parts) if parts else "（无有效结果）"
    blocked_text  = "\n".join(blocked) if blocked else "无"

    user_content = (
        f"用户原始问题：{refined_query}\n\n"
        f"收集到的子任务结果：\n{evidence_text}\n\n"
        f"阻塞任务（根据现有资料无法回答的部分）：\n{blocked_text}"
    )

    system_prompt = _FINAL_SKILL_PATH.read_text(encoding="utf-8")

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])

    answer = response.content.strip() if response.content else "（LLM 返回了空响应）"

    for field in ("people_id", "event_id", "chunk_id"):
        valid_ids = Extract_Cited_Ids(evidence_text, field)
        error = Validate_Citations(answer, field, valid_ids)
        if error is not None:
            print(f"\n[Citation 报警] _Synthesize_Final_Answer：{error}")
            return "根据现有资料，无法生成可靠回答。"

    return answer
