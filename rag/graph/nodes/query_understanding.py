"""Query Understanding 节点。

本模块实现 QU 节点，将用户原始问题解析为结构化检索计划：
    _Load_Skill()               — 读取 query_understanding.md 作为 system prompt。
    _Validate_Query_Plan(plan)  — 校验 LLM 输出的 JSON，返回错误描述或 None。
    Query_Understanding_Node()  — 调用 LLM，校验结果，最多重试 2 次。
"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from rag.config.settings import get_llm


_SKILL_PATH = (
    Path(__file__).resolve().parents[2] / "agent" / "skills" / "query_understanding.md"
)

_VALID_QUERY_KIND = {"fact", "analysis", "multi_enum"}
_VALID_INTENTION  = {"people", "timeline", "direct"}


def _Load_Skill() -> str:
    return _SKILL_PATH.read_text(encoding = "utf-8")


def _Validate_Query_Plan(plan: dict) -> str | None:
    """校验 LLM 输出的查询计划字典，返回错误描述字符串或 None（通过）。

    校验规则表：
    | 字段           | 规则                                                        |
    |----------------|-------------------------------------------------------------|
    | refined_query  | 必须存在，非空字符串                                        |
    | task_type      | 只能是 "single" 或 "subtasks"                               |
    | tasks 长度     | single → 恰好 1 条；subtasks → 至少 2 条                   |
    | task_id        | 格式 t{n}，从 t1 开始连续编号，不允许跳号                  |
    | query_kind     | 只能是 "fact" / "analysis" / "multi_enum"                  |
    | intention      | 只能是 "people" / "timeline" / "direct"                    |
    | depends_on     | 引用的 task_id 必须存在且在当前 task 之前出现               |
    """

    if not isinstance(plan.get("refined_query"), str) or not plan["refined_query"].strip():
        return "refined_query 必须是非空字符串"

    task_type = plan.get("task_type")
    if task_type not in ("single", "subtasks"):
        return f"task_type 必须是 'single' 或 'subtasks'，实际得到 {task_type!r}"

    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or len(tasks) == 0:
        return "tasks 必须是非空列表"

    if task_type == "single" and len(tasks) != 1:
        return f"task_type=single 时 tasks 只能有 1 条，实际得到 {len(tasks)} 条"

    if task_type == "subtasks" and len(tasks) < 2:
        return f"task_type=subtasks 时 tasks 至少 2 条，实际得到 {len(tasks)} 条"

    seen_ids = []
    for i, task in enumerate(tasks, 1):
        expected_id = f"t{i}"
        if task.get("task_id") != expected_id:
            return f"第 {i} 条 task 的 task_id 应为 {expected_id!r}，实际得到 {task.get('task_id')!r}"

        if task.get("query_kind") not in _VALID_QUERY_KIND:
            return f"task_id={task.get('task_id')} 的 query_kind {task.get('query_kind')!r} 不合法"

        if task.get("intention") not in _VALID_INTENTION:
            return f"task_id={task.get('task_id')} 的 intention {task.get('intention')!r} 不合法"

        depends_on = task.get("depends_on")
        if not isinstance(depends_on, list):
            return f"task_id={task.get('task_id')} 的 depends_on 必须是列表"

        for dep in depends_on:
            if dep not in seen_ids:
                return f"task_id={task.get('task_id')} 的 depends_on 引用了不存在或尚未出现的 {dep!r}"

        seen_ids.append(task["task_id"])

    return None


def Query_Understanding_Node(raw_query: str, history: list = []) -> dict:
    """解析用户问题，输出结构化查询计划。

    加载 query_understanding.md 作为 system prompt，用 json_object 模式约束 LLM 输出。
    对结果做 schema 校验，校验失败时把错误描述追加到对话让 LLM 自我修正，最多重试 2 次。

    Args:
        raw_query: 用户原始问题。
        history: 多轮对话历史，每条含 role / content，为空时跳过指代消解。
    Returns:
        通过校验的查询计划字典，含 refined_query / task_type / tasks。
    Raises:
        ValueError: 3 次尝试均未通过校验。
    """

    system_prompt = _Load_Skill()
    payload = json.dumps({"raw_query": raw_query, "history": history}, ensure_ascii = False)

    llm_json = get_llm().bind(response_format = {"type": "json_object"})

    messages = [
        SystemMessage(content = system_prompt),
        HumanMessage(content = payload),
    ]

    error = None
    for _ in range(3):
        response = llm_json.invoke(messages)
        raw_text = response.content

        try:
            plan = json.loads(raw_text)
        except json.JSONDecodeError as e:
            error = f"输出不是合法 JSON：{e}"
            plan = None
        else:
            error = _Validate_Query_Plan(plan)

        if error is None:
            return plan

        messages.append(response)
        messages.append(
            HumanMessage(content = f"你的输出有以下问题，请严格按格式重新输出：{error}")
        )

    raise ValueError(f"Query_Understanding_Node 在 3 次尝试后仍未输出合法计划，最后错误：{error}")