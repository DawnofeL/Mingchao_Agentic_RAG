"""People 路由节点。

本模块实现 LangGraph 的 route_people 节点，负责以下步骤：
    1. 读取 people_plan.md 作为 system prompt。
    2. 将 PEOPLE_TOOLS 注册给 LLM，发起 native tool call。
    3. 打印 [Tool Used] 工具名和参数 JSON。
    4. 执行 LLM 选定的工具，打印 [Tool Result]（日志截断 300 字，LLM 拿最多 ~30000 token）。
    5. 把结果回流给第二次 LLM，动态绑定"另一个工具 + check_chunk"（强制换工具）：
       - LLM 输出文本 → 返回文本答案。
       - LLM 调用另一个工具 → 执行后进入第三次 LLM 最终判断。
       - LLM 调用 check_chunk 且工具有结果 → 提取 partial answer，返回 "__SUPPLEMENT__" + partial，
         由上游合并 chunk 兜底结果一起回答。
       - LLM 调用 check_chunk 且工具无结果 → 返回 None，由上游纯 chunk 兜底。

调用关系：
    Route_People_Node → PEOPLE_TOOLS（people_tools.py）→ People_Search / Relationships_Search（people_store.py）
"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from rag.config.settings import get_llm
from rag.graph.citation_check import Validate_Citations
from rag.retrieval.tools.people_tools import CHECK_CHUNK_TOOL, PEOPLE_TOOLS


_SKILL_PATH = (
    Path(__file__).resolve().parents[2] / "agent" / "skills" / "people_plan.md"
)
_TOOL_MAP       = {t.name: t for t in PEOPLE_TOOLS}

_LOG_TRUNCATE    = 300
_MAX_RESULT_CHARS = 90_000  # ~30000 token 兜底（中文+JSON 约 3 chars/token）


def _Clip(result: object) -> str:
    text = str(result)
    if len(text) <= _MAX_RESULT_CHARS:
        return text
    return text[:_MAX_RESULT_CHARS] + "\n...（结果过长，已截断至约 30000 token）"


def _Load_Skill() -> str:
    return _SKILL_PATH.read_text(encoding = "utf-8")


_SUPPLEMENT   = "__SUPPLEMENT__"

_LIST_FIELDS = {"entities", "era_filter"}

def _Sanitize_Args(args: dict) -> dict:
    """空字符串 → None；list 字段若 LLM 传成 JSON 字符串则反序列化；单值自动包成列表。

    LLM 有时把可选参数填成 "" 而不是 null，或把 list 参数序列化成字符串传入，
    或把列表参数填成单个值（如 era_filter="洪武" 而非 ["洪武"]）。
    统一在执行前清洗，保证参数类型与工具签名一致。

    Args:
        args: LLM 输出的工具参数字典。
    Returns:
        清洗后的参数字典，原字典不被修改。
    """

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
        elif isinstance(v, str):
            result[k] = v.strip('"')
        else:
            result[k] = v
    return result


def _Execute_Tool(tool_name: str, args: dict) -> object:
    """根据工具名执行对应工具，返回原始结果。

    Args:
        tool_name: LLM 选定的工具名。
        args: LLM 填写的工具参数字典（执行前会先做空字符串清洗）。
    Returns:
        工具执行的原始返回值，类型由各工具决定。
    """

    tool_fn = _TOOL_MAP.get(tool_name)

    if tool_fn is None:
        return f"【未知工具】{tool_name}，只能使用 people_search 或 relationships_search。"

    return tool_fn.invoke(_Sanitize_Args(args))


def _Log_Tool_Used(tool_name: str, args: dict) -> None:
    args_json = json.dumps(args, ensure_ascii = False, indent = 2)
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
        "请先列出你从工具结果中已能确认的内容（带 [people_id=N] 引用），"
        "只列已确认条目，不展开、不推断。若工具结果为空则只回复：无。"
    )
    resp = get_llm().invoke(messages + [HumanMessage(content=partial_prompt)])
    return (resp.content or "").strip()


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


def _Log_Agent_Result(conclusion: str) -> None:
    # 答案改走返回值与 event:answer，日志不再打印答案正文，只在空响应时留个提示
    if not conclusion or not conclusion.strip():
        print("\n[People] LLM 返回了空响应，无法给出结论")


_CITATION_REJECT = "根据现有资料，无法回答此部分。"


def _Ids_From_Result(tool_name: str, tool_result: object) -> set[int]:
    """从工具执行结果里收集合法 people_id，供答案引用校验用。

    people_search 返回列表，每条记录自带 people_id；
    relationships_search 返回字典，只有查询主体自己的 people_id 合法，
    relationships[].target 是人名，没有独立 id，不计入合法集合。

    Args:
        tool_name: 刚执行完的工具名。
        tool_result: 该工具的原始返回值。
    Returns:
        本次工具调用贡献的合法 people_id 集合。
    """

    if tool_name == "people_search" and isinstance(tool_result, list):
        return {
            r["people_id"] for r in tool_result
            if isinstance(r, dict) and isinstance(r.get("people_id"), int)
        }

    if tool_name == "relationships_search" and isinstance(tool_result, dict):
        pid = tool_result.get("people_id")
        return {pid} if isinstance(pid, int) else set()

    return set()


def _Guard_Citations(text: str, valid_ids: set[int], step: str) -> str:
    """校验答案里的 people_id 引用，不合法就打日志报警并改成拒答文案。

    Args:
        text: 待校验的答案文本。
        valid_ids: 当前已执行的工具调用贡献的合法 people_id 集合。
        step: 当前所在步骤名，仅用于日志定位。
    Returns:
        校验通过原样返回 text；不通过返回固定拒答文案。
    """

    error = Validate_Citations(text, "people_id", valid_ids)
    if error is None:
        return text

    print(f"\n[Citation 报警] {step}：{error}")
    return _CITATION_REJECT


def Route_People_Node(task_text: str) -> str | None:
    """People 路由节点主函数。

    用 people_plan.md 作为 system prompt，让 LLM 选择工具并填写参数
    （一次机会，不循环），执行工具后把完整结果回流给第二次 LLM 做判断。
    第二次 LLM 绑定 check_chunk 信号工具：调用即代表无法回答，需 chunk 兜底。

    Args:
        task_text: 当前 task 的问题文本。
    Returns:
        - 字符串（不含 sentinel）：LLM 给出的完整文本答案。
        - "__SUPPLEMENT__" + partial：工具有结果但不完整，上游需合并 chunk 一起回答。
        - None：工具无结果，由上游纯 chunk 兜底。
    """

    system_prompt   = _Load_Skill()
    _llm            = get_llm()
    _LLM_WITH_TOOLS = _llm.bind_tools(PEOPLE_TOOLS)
    _LLM_WITH_CHECK = _llm.bind_tools([CHECK_CHUNK_TOOL])

    messages = [
        SystemMessage(content = system_prompt),
        HumanMessage(content = task_text),
    ]

    # 第一次 LLM 调用：选择工具并填写参数
    response = _LLM_WITH_TOOLS.invoke(messages)

    # LLM 未发出 native tool call：尝试从文本 JSON 降级解析，否则直接返回内容
    if not response.tool_calls:
        parsed = _Parse_Text_Tool_Call(response.content)
        if not parsed:
            text = _Guard_Citations(response.content, set(), "未调用工具直接作答")
            _Log_Agent_Result(text)
            return text
        print("[People] LLM 未触发 native tool call，从文本 JSON 降级解析。")
        tool_name, tool_args = parsed
        response.tool_calls = [{"name": tool_name, "args": tool_args, "id": "text_fallback"}]

    tool_call   = response.tool_calls[0]
    tool_name   = tool_call["name"]
    tool_args   = tool_call["args"]
    tool_result = _Execute_Tool(tool_name, tool_args)

    _Log_Tool_Used(tool_name, tool_args)
    _Log_Tool_Result(tool_result)

    valid_people_ids = _Ids_From_Result(tool_name, tool_result)

    messages.append(response)
    messages.append(
        ToolMessage(
            content      = _Clip(tool_result),
            tool_call_id = tool_call["id"],
        )
    )

    # 第二次 LLM 调用：只绑定另一个工具 + check_chunk，并在提示中明确禁止重复调用
    other_tools  = [t for t in PEOPLE_TOOLS if t.name != tool_name]
    other_names  = " / ".join(t.name for t in other_tools)
    llm_retry    = get_llm().bind_tools(other_tools + [CHECK_CHUNK_TOOL])
    retry_prompt = (
        f"你刚才调用了 {tool_name}，结果已在上方。"
        f"请判断结果能否回答 task：\n"
        f"· 能回答 → 直接输出文本结论，每处引用标 [people_id=N]，严禁展开。\n"
        f"· 不能回答 → 必须调用 {other_names} 补充检索，或调用 check_chunk 触发兜底。\n"
        f"严禁再次调用 {tool_name}。"
    )
    messages.append(HumanMessage(content=retry_prompt))
    judgment = llm_retry.invoke(messages)

    # 直接给出文本答案
    if not judgment.tool_calls:
        text = _Guard_Citations(judgment.content, valid_people_ids, "第二次判断直接给出答案")
        _Log_Agent_Result(text)
        return text

    # 调用 check_chunk：提取 partial 后交上游合并 chunk
    if any(tc["name"] == "check_chunk" for tc in judgment.tool_calls):
        partial = _Generate_Partial(messages)
        if partial and not partial.startswith("无"):
            partial = _Guard_Citations(partial, valid_people_ids, "partial 摘要（首次工具后）")
            if partial == _CITATION_REJECT:
                return _CITATION_REJECT
            print("\n[Agent Result] 工具结果不完整，将与 chunk 合并回答。")
            return _SUPPLEMENT + partial
        print("\n[Agent Result] 工具无结果，回退至 chunk 兜底检索。")
        return None

    # 调用了另一个工具：执行后进行最终判断
    retry_call   = judgment.tool_calls[0]
    retry_name   = retry_call["name"]
    retry_args   = retry_call["args"]
    retry_result = _Execute_Tool(retry_name, retry_args)

    print("\n[Retry Tool]")
    _Log_Tool_Used(retry_name, retry_args)
    _Log_Tool_Result(retry_result)

    valid_people_ids = valid_people_ids | _Ids_From_Result(retry_name, retry_result)

    messages.append(judgment)
    messages.append(
        ToolMessage(
            content      = _Clip(retry_result),
            tool_call_id = retry_call["id"],
        )
    )

    # 第三次 LLM 调用：只绑定 check_chunk，最终判断
    final = _LLM_WITH_CHECK.invoke(messages)

    if final.tool_calls and any(tc["name"] == "check_chunk" for tc in final.tool_calls):
        partial = _Generate_Partial(messages)
        if partial and not partial.startswith("无"):
            partial = _Guard_Citations(partial, valid_people_ids, "partial 摘要（第二次工具后）")
            if partial == _CITATION_REJECT:
                return _CITATION_REJECT
            print("\n[Agent Result] 两次工具结果不完整，将与 chunk 合并回答。")
            return _SUPPLEMENT + partial
        print("\n[Agent Result] 两次工具均无结果，回退至 chunk 兜底检索。")
        return None

    text = _Guard_Citations(final.content, valid_people_ids, "第三次最终判断")
    _Log_Agent_Result(text)
    return text
