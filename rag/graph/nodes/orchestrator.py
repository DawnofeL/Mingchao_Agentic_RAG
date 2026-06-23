"""Orchestrator 编排子图。

本模块把多子任务的拓扑调度实现成一张有环的 LangGraph 子图，只在 task_type == "subtasks"
时介入。核心是 orchestrator-worker 循环：orchestrator 算出本轮 ready 任务并用 Send 扇出，
worker 并行执行后把结果合并回 pool，再回到 orchestrator 算下一轮，直到没有待办任务。

节点：
    orchestrator — 算 ready 任务、解引用 / 枚举增生、写阻塞结果，产出本轮 jobs。
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


# ── 纯函数：节点真正调用的逻辑，跟 LangGraph 本身无关 ───────────────────

def _Parse_Resolved(raw: str) -> list[str]:
    """从 _Resolve_References 拿到的 LLM 原始输出 raw 里提取 resolved_tasks 字段，
       按解析结果分三种情况：
        ① raw 整段就是规整 JSON，直接 json.loads 解析出来，取 resolved_tasks 列表返回。
        ② 整段解析失败（说明 LLM 在 JSON 前后夹了多余文字），用正则抠出最外层
           {...} 子串再解析一次，成功就返回对应 resolved_tasks 列表。
        ③ 两次解析都失败，或者解析出来 resolved_tasks 不存在/不是字符串，
           返回空列表，交给调用方 _Resolve_References 按"上游不足"处理。

    Args:
        raw: LLM 返回内容去掉首尾空白后的原始文本，预期是个 JSON 字符串。
    Returns:
        resolved_tasks 列表，元素是去除空白后的非空字符串；解析失败时返回 []。
    """

    # 先当整段 raw 就是规整 JSON，直接解析取 resolved_tasks
    try:
        data = json.loads(raw)
        return [s for s in data.get("resolved_tasks", []) if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, AttributeError):
        pass

    # 整段解析失败，说明 LLM 输出里夹了多余文字，正则抠出最外层 {...} 再试一次
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return [s for s in data.get("resolved_tasks", []) if isinstance(s, str) and s.strip()]
        except (json.JSONDecodeError, AttributeError):
            pass

    # 两次都解析不出来，返回空列表，上游按"上游不足/阻塞"处理
    return []


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

    # 把每个依赖任务在 pool 里已有的答案拼成文本，作为 LLM 解引用时的上游依据
    upstream_parts = []
    for dep_id in downstream["depends_on"]:
        results = pool.get(dep_id, [])
        answers = "\n".join(r.answer for r in results)
        upstream_parts.append(f"上游任务 {dep_id}：\n{answers}")

    # 拼成最终喂给 LLM 的用户输入：原始问题 + 上游答案 + 下游待执行任务本身
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

    # resolved 是 _Parse_Resolved 解析出来的 resolved_tasks 列表，空/单条/多条三种含义见上面 Returns
    raw = response.content.strip() if response.content else ""
    resolved = _Parse_Resolved(raw)
    print(f"[Resolve] 输出：{resolved}")
    return resolved


def _Execute_Task(parent: dict, resolved_text: str, history: list, top_k: int = 10) -> TaskResult:
    """_Node_Worker 拿到一个 job 后真正执行子任务的地方，按 parent["intention"]
       分四种情况派发到对应的 _Run_* 执行器：
        ① intention == "people"   → 调 _Run_People，按人物相关问题处理。
        ② intention == "timeline" → 调 _Run_Timeline，按时间线相关问题处理。
        ③ intention == "direct"  → 调 _Run_Direct，结合 history 直接对话式回答。
        ④ 其余未知 intention      → 不报错，answer 直接写成提示字符串占位。
       拿到 answer 后统一包成 TaskResult 返回，answer 为空时兜底成"无法回答"提示。

    延迟导入 route_task 的 _Run_* 函数，避免与 route_task.py 形成模块级循环引用。

    Args:
        parent:        原始 task_item 字典，提供 intention / task_id / query_kind。
        resolved_text: 已经过 _Resolve_References 解引用后的、可直接执行的问题文本。
        history:       多轮对话历史，只有 intention == "direct" 时会用到。
        top_k:         people/timeline 检索时取的候选数量。
    Returns:
        TaskResult，answer 字段是这次子任务的回答文本。
    """

    from rag.graph.nodes.route_task import _Run_Direct, _Run_People, _Run_Timeline

    intention  = parent["intention"]
    task_id    = parent["task_id"]
    query_kind = parent.get("query_kind", "fact")
    preview    = resolved_text[:60] + ("…" if len(resolved_text) > 60 else "")
    print(f"\n[Task] {task_id} ({intention}) → {preview}")

    # intention 决定走哪条执行链路，跟 route_task.py 里 Synthesize_Answer 上游用的是同一套 _Run_* 函数
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

    # 按 task_id 排序遍历结果池，固定顺序方便日志和复现问题
    for task_id, results in sorted(pool.items()):
        for r in results:

            # blocked 任务单独收集成提示文本，不混进证据文本，避免 LLM 误把占位回答当真实结果
            if r.blocked:
                blocked.append(f"- {task_id}：\"{r.task}\"")
            else:
                parts.append(f"[{task_id}] 任务：\"{r.task}\"\n     回答：{r.answer}")

    # 没有任何正常结果/阻塞结果时分别给兜底文案，避免拼出空段落
    evidence_text = "\n\n".join(parts) if parts else "（无有效结果）"
    blocked_text  = "\n".join(blocked) if blocked else "无"

    # 拼成最终喂给 LLM 的用户输入：原始问题 + 证据文本 + 阻塞任务提示
    user_content = (
        f"用户原始问题：{refined_query}\n\n"
        f"收集到的子任务结果：\n{evidence_text}\n\n"
        f"阻塞任务（根据现有资料无法回答的部分）：\n{blocked_text}"
    )

    # system_prompt 来自 final_answer.md 这份 skill 文件，规定了合成规则和引用格式
    system_prompt = _FINAL_SKILL_PATH.read_text(encoding="utf-8")

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])

    # response.content 为空时给兜底文案，避免后面引用校验对着 None 取值报错
    answer = response.content.strip() if response.content else "（LLM 返回了空响应）"

    # 逐字段校验最终答案里的引用 id 是不是都来自 evidence_text 本身已出现过的 id，
    # 防止 LLM 合成时编造了证据里没有的引用
    for field in ("people_id", "event_id", "chunk_id"):
        valid_ids = Extract_Cited_Ids(evidence_text, field)
        error = Validate_Citations(answer, field, valid_ids)
        if error is not None:
            print(f"\n[Citation 报警] _Synthesize_Final_Answer：{error}")
            return "根据现有资料，无法生成可靠回答。"

    return answer


# ── LangGraph 编排子图：orchestrator-worker 循环 + Send 扇出 ───────────────────

def _Node_Orchestrator(state: OrchestratorState) -> dict:
    """算出本轮 ready 任务，解引用 / 枚举增生 / 写阻塞结果，产出本轮 jobs。
       route 字段告诉后面的路由怎么走：
        "dispatch"   → 有 job，扇出给 worker 并行执行
        "loop"       → 本轮全阻塞但还有 pending，回到 orchestrator 算下一轮
        "synthesize" → 死锁或全部完成，进入终答合成
    """

    pending   = state["pending"]
    pool      = state["pool"]
    refined_q = state["refined_query"]

    # pending 是空列表，说明所有子任务都已经跑完进了 pool，没活可派了，直接进合成
    if not pending:
        return {"jobs": [], "route": "synthesize"}

    round_num = state["round_num"] + 1

    # 依赖的任务 id 都能在 pool 里找到，才说明依赖全跑完了，这个任务才算 ready
    ready = [t for t in pending if all(dep in pool for dep in t["depends_on"])]

    # pending 还有任务，但 ready 是空的，说明剩下的任务全在互相等依赖，谁都跑不了，判定死锁
    if not ready:
        print(f"\n[Orchestrator] 调度死锁，剩余：{[t['task_id'] for t in pending]}")
        return {"round_num": round_num, "jobs": [], "route": "synthesize"}

    # ready_ids 是这一轮要跑的任务 id 集合，new_pending 是从 pending 里把这些挑出去之后剩下的
    ready_ids   = {t["task_id"] for t in ready}
    new_pending = [t for t in pending if t["task_id"] not in ready_ids]

    print(f"\n[Orchestrator] 第 {round_num} 轮  ready = {[t['task_id'] for t in ready]}")

    jobs:         list[dict] = []
    blocked_pool: dict       = {}

    for t in ready:

        # 没有依赖的任务不用解引用，task 原文直接就是要执行的文本
        if not t["depends_on"]:
            jobs.append({"parent": t, "text": t["task"]})
            continue

        # 有依赖的任务要先把依赖结果代进去，_Resolve_References 处理指代还原/枚举增生
        resolved = _Resolve_References(t, pool, refined_q)

        # resolved 是空列表，说明依赖的上游结果不够支撑这个任务，直接判定阻塞，塞进 blocked_pool
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

        # resolved 只有一条，是单纯的指代还原；多条说明一个任务被拆成了多条枚举子任务
        if len(resolved) == 1:
            print(f"[Orchestrator] {t['task_id']} 指代还原 → {resolved[0]}")
        else:
            print(f"[Orchestrator] {t['task_id']} 枚举增生 → {len(resolved)} 条")
            for i, text in enumerate(resolved, 1):
                print(f"  [{i}] {text}")

        # resolved 里每条文本各自变成一个 job，挂在同一个 parent 任务下面
        for text in resolved:
            jobs.append({"parent": t, "text": text})

    updates: dict = {"round_num": round_num, "pending": new_pending, "jobs": jobs}
    if blocked_pool:
        updates["pool"] = blocked_pool   # 经 merge_pool reducer 合并

    # jobs 有内容就扇出给 worker 并行跑；jobs 为空但 new_pending 还有任务，说明这轮全阻塞，
    # 回 orchestrator 算下一轮；jobs 和 new_pending 都空，才是真的没活干了，进合成
    if jobs:
        print(f"\n[Orchestrator] 并发执行 {len(jobs)} 个 job")
        updates["route"] = "dispatch"
    elif new_pending:
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
    """"dispatch"/"loop"/"synthesize" 不是 LangGraph 的内置概念，是 _Node_Orchestrator
       自己定义的三个普通字符串，写进 state["route"]，本函数只是读出来翻译成实际动作：
        ① route == "dispatch"   → 有 job 要跑，包成 Send 列表，扇出给多个 worker 并行执行。
        ② route == "loop"       → 这轮全阻塞但还有没跑的任务，返回 "orchestrator" 回去重算一轮。
        ③ route == "synthesize" → 没活干了（跑完或死锁），返回 "synthesize" 进合成节点。

    Args:
        state: 当前 OrchestratorState，读 route / jobs / history / top_k 字段。
    Returns:
        ① 时返回 Send 对象列表（扇出多个 worker）；②③ 时返回目标节点名字符串。
    """

    route = state["route"]

    if route == "dispatch":

        # 每个 job 拼成一份 payload 交给 worker，parent 是原始 task_item，
        # text 是已解引用的执行文本，history/top_k 是全局共享字段
        return [
            Send("worker", {
                "parent":        job["parent"],
                "resolved_text": job["text"],
                "history":       state["history"],
                "top_k":         state["top_k"],
            })
            for job in state["jobs"]
        ]

    # 本轮没派出任何 job，但 pending 里还有任务，说明全阻塞了，回 orchestrator 重新算
    if route == "loop":
        return "orchestrator"

    # 既不是 dispatch 也不是 loop，说明没活可干了，进入终答合成
    return "synthesize"


# ── 图的搭建与入口 ───────────────────

def _Build_Orchestrator_Graph():
    """组装并编译多子任务编排子图，举例说明 _Route_From_Orchestrator 的三种走向：
        dispatch   — task_1 没依赖，直接能跑，扇出给 worker。
        loop       — task_2 依赖 task_1，但 task_1 答案不够用，task_2 被判阻塞；
                      这轮没活干但 task_3 还没跑，回去再算一轮。
        synthesize — 所有任务都跑完了，或者剩下的任务互相卡死谁都跑不了，
                      直接进去合成答案。
    """

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
