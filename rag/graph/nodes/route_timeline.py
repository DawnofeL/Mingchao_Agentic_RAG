"""Timeline 路由子图。

本模块把 timeline 检索的两段式工具调用逻辑实现成一张 LangGraph 子图，节点如下：
    first_call   — 第一次 LLM 调用，绑定 TIMELINE_TOOLS 填 event_search 参数（或直接作答）。
    execute_tool — 执行 event_search，累计合法 event_id，结果回填 messages。
    judge        — 第二次 LLM 调用，绑定 check_chunk 二选一判断（空响应强制兜底）。
    make_partial — check_chunk 触发后提取 partial，决定 supplement 还是纯 chunk 兜底。

子图入口 Route_Timeline_Node 保持原有返回契约不变（str / "__SUPPLEMENT__"+partial / None），
上游 route_task._Run_Timeline 无需改动。

调用关系：
    Route_Timeline_Node → timeline 子图 → TIMELINE_TOOLS（timeline_tools.py）
                        → Event_Search（timeline_store.py）
"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from rag.config.settings import get_llm
from rag.graph.citation_check import Validate_Citations
from rag.graph.state import RetrievalState
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


_CITATION_REJECT = "根据现有资料，无法回答此部分。"


def _Ids_From_Result(tool_result: object) -> set[int]:
    """从 event_search 返回结果里收集合法 event_id，供答案引用校验用。"""

    if isinstance(tool_result, list):
        return {
            e["event_id"] for e in tool_result
            if isinstance(e, dict) and isinstance(e.get("event_id"), int)
        }
    return set()


def _Guard_Citations(text: str, valid_ids: set[int], step: str) -> str:
    """校验答案里的 event_id 引用，不合法就打日志报警并改成拒答文案。"""

    error = Validate_Citations(text, "event_id", valid_ids)
    if error is None:
        return text

    print(f"\n[Citation 报警] {step}：{error}")
    return _CITATION_REJECT


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


# ── LangGraph 子图：把两段式工具调用拆成节点 + 条件边 ─────────────────────────

def _Node_First_Call(state: RetrievalState) -> dict:
    """第一次 LLM 调用：绑定 TIMELINE_TOOLS 填 event_search 参数。

    未触发 native tool call 时先尝试文本 JSON 降级解析，仍失败就当作直接文本答案收尾。
    """

    messages = [
        SystemMessage(content = _Load_Skill()),
        HumanMessage(content = state["task_text"]),
    ]

    response = get_llm().bind_tools(TIMELINE_TOOLS).invoke(messages)

    if not response.tool_calls:
        parsed = _Parse_Text_Tool_Call(response.content)
        if not parsed:
            text = _Guard_Citations(response.content, set(), "未调用工具直接作答")
            _Log_Agent_Result(text)
            return {"messages": messages, "result_kind": "answer", "answer": text}
        print("[Timeline] LLM 未触发 native tool call，从文本 JSON 降级解析。")
        tool_name, tool_args = parsed
        response.tool_calls = [{"name": tool_name, "args": tool_args, "id": "text_fallback"}]

    call = response.tool_calls[0]
    return {
        "messages":     messages + [response],
        "pending_tool": {"name": call["name"], "args": call["args"], "id": call["id"]},
    }


def _Node_Execute_Tool(state: RetrievalState) -> dict:
    """执行 event_search，累计合法 event_id，把结果回填进 messages。"""

    call   = state["pending_tool"]
    result = _Execute_Tool(call["name"], call["args"])

    _Log_Tool_Used(call["name"], call["args"])
    _Log_Tool_Result(result)

    messages = state["messages"] + [
        ToolMessage(content = _Clip(result), tool_call_id = call["id"])
    ]
    return {
        "messages":  messages,
        "valid_ids": state["valid_ids"] | _Ids_From_Result(result),
    }


def _Node_Judge(state: RetrievalState) -> dict:
    """第二次 LLM 调用：绑定 check_chunk 二选一判断，空响应强制兜底。"""

    retry_prompt = (
        "工具结果已在上方。请二选一：\n"
        "· 能完整回答 task → 直接输出文本结论，每处引用标 [event_id=N]，严禁展开。\n"
        "· 结果不足以回答（为空 / 缺失关键事件 / 算不出时间差） → 必须调用 check_chunk 触发兜底。\n"
        "严禁返回空响应或解释性文字。"
    )

    messages = state["messages"] + [HumanMessage(content = retry_prompt)]
    judgment = get_llm().bind_tools([CHECK_CHUNK_TOOL]).invoke(messages)

    if judgment.tool_calls and any(tc["name"] == "check_chunk" for tc in judgment.tool_calls):
        return {"messages": messages, "result_kind": "check"}

    # 代码兜底：LLM 没听话返空响应时，强制走 chunk
    if not judgment.content or not judgment.content.strip():
        print("\n[Agent Result] LLM 返空响应，强制回退至 chunk 兜底检索。")
        return {"messages": messages, "result_kind": "fallback"}

    text = _Guard_Citations(judgment.content, state["valid_ids"], "第二次判断直接给出答案")
    _Log_Agent_Result(text)
    return {"messages": messages, "result_kind": "answer", "answer": text}


def _Node_Make_Partial(state: RetrievalState) -> dict:
    """check_chunk 触发后提取 partial，决定走 supplement 合并还是纯 chunk 兜底。"""

    partial = _Generate_Partial(state["messages"])

    if partial and not partial.startswith("无"):
        partial = _Guard_Citations(partial, state["valid_ids"], "partial 摘要")
        if partial == _CITATION_REJECT:
            return {"result_kind": "answer", "answer": _CITATION_REJECT}
        print("\n[Agent Result] 工具结果不完整，将与 chunk 合并回答。")
        return {"result_kind": "supplement", "answer": partial}

    print("\n[Agent Result] 工具无结果，回退至 chunk 兜底检索。")
    return {"result_kind": "fallback"}


def _Route_After_First(state: RetrievalState) -> str:
    return "end" if state.get("result_kind") == "answer" else "execute"


def _Route_After_Judge(state: RetrievalState) -> str:
    return "partial" if state["result_kind"] == "check" else "end"


def _Build_Timeline_Graph():
    """组装并编译 timeline 检索子图。"""

    builder = StateGraph(RetrievalState)

    builder.add_node("first_call",   _Node_First_Call)
    builder.add_node("execute_tool", _Node_Execute_Tool)
    builder.add_node("judge",        _Node_Judge)
    builder.add_node("make_partial", _Node_Make_Partial)

    builder.add_edge(START, "first_call")
    builder.add_conditional_edges(
        "first_call", _Route_After_First,
        {"execute": "execute_tool", "end": END},
    )
    builder.add_edge("execute_tool", "judge")
    builder.add_conditional_edges(
        "judge", _Route_After_Judge,
        {"partial": "make_partial", "end": END},
    )
    builder.add_edge("make_partial", END)

    return builder.compile()


# 模块加载时编译子图，Route_Timeline_Node 直接调用
_timeline_graph = _Build_Timeline_Graph()


def Route_Timeline_Node(task_text: str) -> str | None:
    """Timeline 路由子图入口，保持原有返回契约不变。

    构造初始状态后调用编译好的子图，把终态 result_kind 映射回原契约：
        "answer"     → 文本答案字符串（含直接作答、引用拒答）。
        "supplement" → "__SUPPLEMENT__" + partial，上游合并 chunk 一起回答。
        "fallback"   → None，工具无结果或 LLM 返空响应，由上游纯 chunk 兜底。

    Args:
        task_text: 当前 task 的问题文本。
    Returns:
        str / "__SUPPLEMENT__"+partial / None，三种含义同上。
    """

    init: RetrievalState = {
        "task_text":    task_text,
        "messages":     [],
        "pending_tool": {},
        "valid_ids":    set(),
        "tool_round":   0,
        "result_kind":  "",
        "answer":       "",
    }
    final_state = _timeline_graph.invoke(init)

    kind   = final_state.get("result_kind", "")
    answer = final_state.get("answer", "")

    if kind == "supplement":
        return _SUPPLEMENT + answer
    if kind == "fallback":
        return None
    return answer
