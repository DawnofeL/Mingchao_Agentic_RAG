"""Timeline 路由节点。

本模块实现 timeline 路由节点，负责以下步骤：
    1. 读取 timeline_plan.md 作为 system prompt。
    2. 将 TIMELINE_TOOLS 注册给 LLM，发起 native tool call。
    3. 打印 [Tool Used] 工具名和参数 JSON。
    4. 执行 LLM 选定的工具，打印 [Tool Result]（日志截断 300 字，LLM 拿最多 ~30000 token）。
    5. 第二次 LLM 调用前注入二选一指令，绑定 check_chunk 信号工具做最终判断：
       - LLM 输出文本 → 返回文本答案。
       - LLM 调用 check_chunk 且工具有结果 → 提取 partial answer，返回 "__SUPPLEMENT__" + partial，
         由上游合并 chunk 兜底结果一起回答。
       - LLM 调用 check_chunk 且工具无结果 → 返回 None，由上游纯 chunk 兜底。
       - LLM 返空响应 → 代码强制视同 check_chunk，返回 None，防止 silent failure。

调用关系：
    Route_Timeline_Node → TIMELINE_TOOLS（timeline_tools.py）→ Event_Search（timeline_store.py）
"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from rag.config.settings import get_llm
from rag.retrieval.tools.people_tools import CHECK_CHUNK_TOOL
from rag.retrieval.tools.timeline_tools import TIMELINE_TOOLS


_SKILL_PATH = (
    Path(__file__).resolve().parents[2] / "agent" / "skills" / "timeline_plan.md"
)
_TOOL_MAP       = {t.name: t for t in TIMELINE_TOOLS}

_LOG_TRUNCATE    = 300
_MAX_RESULT_CHARS = 90_000  # ~30000 token 兜底（中文+JSON 约 3 chars/token）


def _Clip(result: object) -> str:
    text = str(result)
    if len(text) <= _MAX_RESULT_CHARS:
        return text
    return text[:_MAX_RESULT_CHARS] + "\n...（结果过长，已截断至约 30000 token）"


def _Load_Skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8")


_SUPPLEMENT  = "__SUPPLEMENT__"

_LIST_FIELDS = {"event_keywords", "participants", "era", "year"}

def _Sanitize_Args(args: dict) -> dict:
    """空字符串 → None；list 字段若 LLM 传成 JSON 字符串则反序列化；单值自动包成列表。"""
    result = {}
    for k, v in args.items():
        if v == "" or v is None:
            result[k] = None
        elif k in _LIST_FIELDS and isinstance(v, str):
            try:
                parsed = json.loads(v)
                result[k] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, ValueError):
                result[k] = [v]
        elif k in _LIST_FIELDS and not isinstance(v, list):
            result[k] = [v]
        else:
            result[k] = v
    return result


def _Execute_Tool(tool_name: str, args: dict) -> object:
    tool_fn = _TOOL_MAP.get(tool_name)
    if tool_fn is None:
        return f"【未知工具】{tool_name}，只能使用 event_search。"
    return tool_fn.invoke(_Sanitize_Args(args))


def _Log_Tool_Used(tool_name: str, args: dict) -> None:
    args_json = json.dumps(args, ensure_ascii=False, indent=2)
    print(f"[Tool Used]  {tool_name}")
    print(args_json)


def _Log_Tool_Result(result: object) -> None:
    result_str = str(result)
    display = result_str if len(result_str) <= _LOG_TRUNCATE else result_str[:_LOG_TRUNCATE] + "…（截断）"
    print(f"\n[Tool Result]")
    print(display)


def _Generate_Partial(messages: list) -> str:
    """工具有结果但 LLM 认为不完整时，提取已确认部分作为 partial answer。"""
    partial_prompt = (
        "工具结果已在上方，你认为还需要 chunk 补充。"
        "请先列出你从工具结果中已能确认的内容（带 [event_id=N] 引用），"
        "只列已确认条目，不展开、不推断。若工具结果为空则只回复：无。"
    )
    resp = get_llm().invoke(messages + [HumanMessage(content=partial_prompt)])
    return (resp.content or "").strip()


def _Log_Agent_Result(conclusion: str) -> None:
    # 答案改走返回值与 event:answer，日志不再打印答案正文，只在空响应时留个提示
    if not conclusion or not conclusion.strip():
        print("\n[Timeline] LLM 返回了空响应，无法给出结论")


def _Parse_Text_Tool_Call(content: str) -> tuple[str, dict] | None:
    """LLM 未触发 native tool call 时，尝试从文本 JSON 中解析工具名和参数。"""
    text = (content or "").strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    tool_name = data.get("tool") or data.get("name")
    if tool_name not in _TOOL_MAP:
        return None
    args = {k: v for k, v in data.items() if k not in ("tool", "name", "type")}
    return tool_name, args


def Route_Timeline_Node(task_text: str) -> str | None:
    """Timeline 路由节点主函数。

    用 timeline_plan.md 作为 system prompt，让 LLM 填写 event_search 参数
    （一次机会，不循环），执行工具后把完整结果回流给第二次 LLM 做判断。
    第二次 LLM 调用前注入二选一指令，绑定 check_chunk 信号工具：
    调用 check_chunk 或返空响应均触发 chunk 兜底（空响应由代码强制兜底）。

    Args:
        task_text: 当前 task 的问题文本。
    Returns:
        - 字符串（不含 sentinel）：LLM 给出的完整文本答案。
        - "__SUPPLEMENT__" + partial：工具有结果但不完整，上游需合并 chunk 一起回答。
        - None：工具无结果或 LLM 返空响应，由上游纯 chunk 兜底。
    """

    system_prompt   = _Load_Skill()
    _llm            = get_llm()
    _LLM_WITH_TOOLS = _llm.bind_tools(TIMELINE_TOOLS)
    _LLM_WITH_CHECK = _llm.bind_tools([CHECK_CHUNK_TOOL])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_text),
    ]

    response = _LLM_WITH_TOOLS.invoke(messages)

    if not response.tool_calls:
        parsed = _Parse_Text_Tool_Call(response.content)
        if not parsed:
            _Log_Agent_Result(response.content)
            return response.content
        print("[Timeline] LLM 未触发 native tool call，从文本 JSON 降级解析。")
        tool_name, tool_args = parsed
        response.tool_calls = [{"name": tool_name, "args": tool_args, "id": "text_fallback"}]

    tool_call   = response.tool_calls[0]
    tool_name   = tool_call["name"]
    tool_args   = tool_call["args"]
    tool_result = _Execute_Tool(tool_name, tool_args)

    _Log_Tool_Used(tool_name, tool_args)
    _Log_Tool_Result(tool_result)

    messages.append(response)
    messages.append(
        ToolMessage(
            content      = _Clip(tool_result),
            tool_call_id = tool_call["id"],
        )
    )

    # 第二次 LLM 调用前明确二选一，防止 LLM 返空响应导致 silent failure
    retry_prompt = (
        "工具结果已在上方。请二选一：\n"
        "· 能完整回答 task → 直接输出文本结论，每处引用标 [event_id=N]，严禁展开。\n"
        "· 结果不足以回答（为空 / 缺失关键事件 / 算不出时间差） → 必须调用 check_chunk 触发兜底。\n"
        "严禁返回空响应或解释性文字。"
    )
    messages.append(HumanMessage(content = retry_prompt))
    judgment = _LLM_WITH_CHECK.invoke(messages)

    if judgment.tool_calls and any(tc["name"] == "check_chunk" for tc in judgment.tool_calls):
        partial = _Generate_Partial(messages)
        if partial and not partial.startswith("无"):
            print("\n[Agent Result] 工具结果不完整，将与 chunk 合并回答。")
            return _SUPPLEMENT + partial
        print("\n[Agent Result] 工具无结果，回退至 chunk 兜底检索。")
        return None

    # 代码兜底：LLM 没听话返空响应时，强制走 chunk
    if not judgment.content or not judgment.content.strip():
        print("\n[Agent Result] LLM 返空响应，强制回退至 chunk 兜底检索。")
        return None

    _Log_Agent_Result(judgment.content)
    return judgment.content
