"""Orchestrator 调度节点。

本模块实现多子任务的拓扑调度，只在 task_type == "subtasks" 时介入：
    Run_Orchestrator         — 主入口：拓扑循环 + 并发执行 + 结果池聚合
    _Resolve_References      — LLM 调用：指代还原 / 枚举增生 / 阻塞判定
    _Execute_Task            — 按 intention 派发到对应 _Run_* 执行器
    _Synthesize_Final_Answer — LLM 调用：读结果池合成最终回答

调用关系：
    Route_Task（subtasks 分支）→ Run_Orchestrator
        → _Resolve_References（per 有依赖任务，串行）
        → _Execute_Task × N（并发，ThreadPoolExecutor）
        → _Synthesize_Final_Answer（一次，结束时）
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from rag.config.settings import get_llm
from rag.graph.citation_check import Extract_Cited_Ids, Validate_Citations


_RESOLVE_SKILL_PATH = Path(__file__).resolve().parents[2] / "agent" / "skills" / "resolve_references.md"
_FINAL_SKILL_PATH   = Path(__file__).resolve().parents[2] / "agent" / "skills" / "final_answer.md"


@dataclass
class TaskResult:
    task_id:   str
    task:      str
    intention: str
    answer:    str
    blocked:   bool = False


def Run_Orchestrator(plan: dict, history: list = [], top_k: int = 10) -> str:
    """多子任务拓扑调度主入口。

    按 depends_on 关系确定每轮 ready 任务：
    - 无依赖任务直接原文执行
    - 有依赖任务先调 _Resolve_References 做指代还原或枚举增生，再并发执行

    Args:
        plan:    QU 输出的查询计划，含 refined_query / task_type / tasks。
        history: 多轮对话历史，供 direct intention 子任务使用。
    """

    pool: dict[str, list[TaskResult]] = {}
    pending   = list(plan["tasks"])
    refined_q = plan["refined_query"]
    round_num = 0

    while pending:
        round_num += 1

        ready = [t for t in pending if all(dep in pool for dep in t["depends_on"])]
        if not ready:
            print(f"\n[Orchestrator] 调度死锁，剩余：{[t['task_id'] for t in pending]}")
            break

        ready_ids = {t["task_id"] for t in ready}
        pending   = [t for t in pending if t["task_id"] not in ready_ids]

        print(f"\n[Orchestrator] 第 {round_num} 轮  ready = {[t['task_id'] for t in ready]}")

        jobs: list[tuple[dict, str]] = []

        for t in ready:
            if not t["depends_on"]:
                jobs.append((t, t["task"]))
                continue

            resolved = _Resolve_References(t, pool, refined_q)

            if not resolved:
                pool[t["task_id"]] = [TaskResult(
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
                jobs.append((t, text))

        if not jobs:
            continue

        print(f"\n[Orchestrator] 并发执行 {len(jobs)} 个 job")

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_Execute_Task, parent, text, history, top_k)
                for parent, text in jobs
            ]
            results = [f.result() for f in futures]

        for result in results:
            pool.setdefault(result.task_id, []).append(result)

    print(f"\n[Orchestrator] 所有子任务完成，进入终答合成\n")
    return _Synthesize_Final_Answer(refined_q, pool)


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
