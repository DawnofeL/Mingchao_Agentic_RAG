"""People 路由子图。

本模块把 people 检索的三段式工具调用逻辑实现成一张 LangGraph 子图，节点如下：
    first_call    — 第一次 LLM 调用，绑定 PEOPLE_TOOLS 选工具填参数（或直接作答）。
    execute_tool  — 执行选定工具，累计合法 id，结果回填 messages。
    judge         — 第二次 LLM 调用，绑定另一个工具 + check_chunk 判断够不够回答。
    execute_retry — 换用另一个工具补检索。
    final_judge   — 第三次 LLM 调用，只绑定 check_chunk 做最终判断。
    make_partial  — check_chunk 触发后提取 partial，决定 supplement 还是纯 chunk 兜底。

子图入口 Route_People_Node 保持原有返回契约不变（str / "__SUPPLEMENT__"+partial / None），
上游 route_task._Run_People 无需改动。

调用关系：
    Route_People_Node → people 子图 → PEOPLE_TOOLS（people_tools.py）
                      → People_Search / Relationships_Search（people_store.py）
"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from rag.config.settings import get_llm
from rag.graph.citation_check import Validate_Citations
from rag.graph.state import RetrievalState
from rag.retrieval.tools.people_tools import CHECK_CHUNK_TOOL, PEOPLE_TOOLS


_SKILL_PATH = (
    Path(__file__).resolve().parents[2] / "agent" / "skills" / "people_plan.md"
)
_TOOL_MAP       = {t.name: t for t in PEOPLE_TOOLS}

_LOG_TRUNCATE    = 300   # 控制台日志打印的截断长度
_MAX_RESULT_CHARS = 90_000  # ~30000 token 兜底（中文+JSON 约 3 chars/token）


def _Clip(result: object) -> str:
    # 工具结果转成字符串后塞进 ToolMessage，太长会把上下文撑爆，超过上限就截断
    text = str(result)
    if len(text) <= _MAX_RESULT_CHARS:
        return text
    return text[:_MAX_RESULT_CHARS] + "\n...（结果过长，已截断至约 30000 token）"


def _Load_Skill() -> str:
    # 读 people_plan.md 这份 skill 文档，当作 SystemMessage 喂给 LLM
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

        # 空字符串或 null，统一当作"没填"
        if v == "" or v is None:
            result[k] = None

        # 该是列表的字段被传成了字符串，尝试解析回列表，解析失败就当单值包一层
        elif k in _LIST_FIELDS and isinstance(v, str):
            try:
                parsed = json.loads(v)
                result[k] = parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, ValueError):
                result[k] = [v]

        # 该是列表的字段被传成了单个值，包一层[]
        elif k in _LIST_FIELDS and not isinstance(v, list):
            result[k] = [v]

        # 普通字符串字段，去掉两端多余的引号
        elif isinstance(v, str):
            result[k] = v.strip('"')

        # 其他类型（数字、布尔值等）原样保留
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

    # 比如 tool_name = "relationships_search"，_TOOL_MAP[tool_name] 查出来是
    # relationships_search 这个 @tool 函数对象本身，不是字符串
    tool_fn = _TOOL_MAP.get(tool_name)

    if tool_fn is None:
        return f"【未知工具】{tool_name}，只能使用 people_search 或 relationships_search。"

    # 把清洗后的参数，送入 LLM 这次选定的那个 people 工具
    return tool_fn.invoke(_Sanitize_Args(args))

# 打印本次调用的工具名和参数作为日志
def _Log_Tool_Used(tool_name: str, args: dict) -> None:
    # json.dumps 把字典转成带缩进的字符串，打印出来像
    # {
    #   "person": "朱棣"
    # }
    args_json = json.dumps(args, ensure_ascii = False, indent = 2)
    print(f"[Tool Used]  {tool_name}")
    print(args_json)


def _Log_Tool_Result(result: object) -> None:
    # str(result) 把工具返回的字典/列表转成字符串，超过 300 字就截断
    result_str = str(result)
    display = result_str if len(result_str) <= _LOG_TRUNCATE else result_str[:_LOG_TRUNCATE] + "…（截断）"
    print(f"\n[Tool Result]")
    print(display)


def _Generate_Partial(messages: list) -> str:
    """
    工具有结果但 LLM 认为不完整时，提取已确认部分作为 partial answer。

    触发条件：judge 或 final_judge 阶段模型选了 check_chunk，说明它判断工具结果
    不够回答，result_kind 被设成 "check"，流程才会走到这个函数。
    """

    # 用户原问题、模型之前的工具调用、工具实际返回的结果，加上partial_prompt后送给LLM
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

    # 不是以 { 开头，连 JSON 都不像，直接放弃解析
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    # 字段名可能是 tool 或 name，兼容两种写法
    tool_name = data.get("tool") or data.get("name")
    if tool_name not in _TOOL_MAP:
        return None

    # 剩下的字段就当作工具参数，tool/name/type 是控制字段，不算参数
    args = {k: v for k, v in data.items() if k not in ("tool", "name", "type")}
    return tool_name, args


def _Log_Agent_Result(conclusion: str) -> None:
    """
    conclusion 是空字符串时打一句日志提示，正常有内容时什么都不做。
    答案正文本身不在这里打印，这个函数只负责"模型这轮什么都没说"这一种异常情况报警。
    """

    # conclusion 为空或全是空白字符，才打日志报警；有内容直接放过，不打印
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

    # r 是字典且 r.get("people_id") 是整数才收集为people_id，避免脏数据和 KeyError
    if tool_name == "people_search" and isinstance(tool_result, list):
        return {
            r["people_id"] for r in tool_result
            if isinstance(r, dict) and isinstance(r.get("people_id"), int)
        }

    # tool_result.get("people_id") 取出来是整数才包成集合返回，否则返回空集合
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

    # Validate_Citations 把 text 里的 [people_id=N] 抠出来跟 valid_ids 比对，
    # 全部能对上就返回 None，只要有一个 id 不在 valid_ids 里就返回错误描述字符串
    error = Validate_Citations(text, "people_id", valid_ids)
    if error is None:
        return text

    # error 不是 None，说明出现了编造或者越界的引用，原文本作废，统一换成拒答文案
    print(f"\n[Citation 报警] {step}：{error}")
    return _CITATION_REJECT


# ======= LangGraph 子图：把三段式工具调用拆成节点 + 条件边 ======= 

def _Node_First_Call(state: RetrievalState) -> dict:
    """子图入口节点，第一次 LLM 调用，绑定 PEOPLE_TOOLS 让模型自己选工具填参数。

    走到这里有三种结局：
    ① native tool call 正常触发，取第一条 tool_call 存进 pending_tool，
       交给下一个节点 execute_tool 真正执行，自己只负责"选"不负责"做"。
    ② native tool call 没触发，但模型把调用意图写成了纯文本 JSON，
       _Parse_Text_Tool_Call 把它解析出来后手动拼成 tool_calls，
       当成跟 ① 一样的情况继续往下走。
    ③ 上面两条都没有，说明模型是真的想直接打字回答，不调用任何工具，
       这种情况下没有合法的 people_id（valid_ids 传空集合），
       答案里只要出现引用就会被 _Guard_Citations 判定越界改写成拒答文案，
       result_kind 记为 "answer"，子图到这里就直接结束，不会进 execute_tool。
    """

    messages = [
        SystemMessage(content = _Load_Skill()),
        HumanMessage(content = state["task_text"]),
    ]

    # bind_tools 绑定 PEOPLE_TOOLS，LLM 自己判断要不要调用、调用哪个、参数填什么
    response = get_llm().bind_tools(PEOPLE_TOOLS).invoke(messages)

    if not response.tool_calls:

        # parsed 解析成功是元组，失败是 None
        parsed = _Parse_Text_Tool_Call(response.content)

        # 文本里没藏工具调用，模型就是想直接打字回答
        if not parsed:

            # 没调过工具，传空集合，文本里出现的任何 [people_id=N] 都判定越界
            text = _Guard_Citations(response.content, set(), "未调用工具直接作答")
            _Log_Agent_Result(text)
            return {"messages": messages, "result_kind": "answer", "answer": text}

        # 文本 JSON 解析成功，手动拼一条 tool_calls，后续流程当成正常调用处理
        print("[People] LLM 未触发 native tool call，从文本 JSON 降级解析。")
        tool_name, tool_args = parsed
        response.tool_calls = [{"name": tool_name, "args": tool_args, "id": "text_fallback"}]

    # 只取第一条 tool_call，存进 state 等下一个节点真正执行
    call = response.tool_calls[0]
    return {
        "messages":     messages + [response],
        "pending_tool": {"name": call["name"], "args": call["args"], "id": call["id"]},
    }


def _Node_Execute_Tool(state: RetrievalState) -> dict:
    """执行第一次选定的工具，累计合法 id，把结果回填进 messages。"""

    call   = state["pending_tool"]
    result = _Execute_Tool(call["name"], call["args"])

    _Log_Tool_Used(call["name"], call["args"])
    _Log_Tool_Result(result)

    # ToolMessage 的 tool_call_id 要跟 LLM 那条 tool_call 的 id 对上，
    # 模型才知道这条结果对应它发出的哪次调用
    messages = state["messages"] + [
        ToolMessage(content = _Clip(result), tool_call_id = call["id"])
    ]
    return {
        "messages":   messages,
        "valid_ids":  state["valid_ids"] | _Ids_From_Result(call["name"], result),
        "tool_round": 1,
    }


def _Node_Judge(state: RetrievalState) -> dict:
    """第二次 LLM 调用：绑定另一个工具 + check_chunk，判断够不够回答。

    三种走向写进 result_kind：answer 直接收尾，check 转 make_partial，retry 换工具再查。
    """

    # 已经调过的那个工具不能再调，只把剩下的工具 + check_chunk 绑给这一轮
    first_name  = state["pending_tool"]["name"]
    other_tools = [t for t in PEOPLE_TOOLS if t.name != first_name]
    other_names = " / ".join(t.name for t in other_tools)
    retry_prompt = (
        f"你刚才调用了 {first_name}，结果已在上方。"
        f"请判断结果能否回答 task：\n"
        f"· 能回答 → 直接输出文本结论，每处引用标 [people_id=N]，严禁展开。\n"
        f"· 不能回答 → 必须调用 {other_names} 补充检索，或调用 check_chunk 触发兜底。\n"
        f"严禁再次调用 {first_name}。"
    )

    messages = state["messages"] + [HumanMessage(content = retry_prompt)]
    judgment = get_llm().bind_tools(other_tools + [CHECK_CHUNK_TOOL]).invoke(messages)

    # 没触发任何工具调用，说明模型直接给出文本结论，本节点就此收尾
    if not judgment.tool_calls:
        text = _Guard_Citations(judgment.content, state["valid_ids"], "第二次判断直接给出答案")
        _Log_Agent_Result(text)
        return {"messages": messages, "result_kind": "answer", "answer": text}

    # check_chunk 路径不把 judgment 入栈，让 make_partial 拿到与原实现一致的 messages
    if any(tc["name"] == "check_chunk" for tc in judgment.tool_calls):
        return {"messages": messages, "result_kind": "check"}

    # 走到这里说明模型选了 other_tools 里的另一个工具，记下来等下一节点执行
    call = judgment.tool_calls[0]
    return {
        "messages":     messages + [judgment],
        "pending_tool": {"name": call["name"], "args": call["args"], "id": call["id"]},
        "result_kind":  "retry",
    }


def _Node_Execute_Retry(state: RetrievalState) -> dict:
    """执行换用的第二个工具，累计合法 id，结果回填 messages。"""

    call   = state["pending_tool"]
    result = _Execute_Tool(call["name"], call["args"])

    print("\n[Retry Tool]")
    _Log_Tool_Used(call["name"], call["args"])
    _Log_Tool_Result(result)

    messages = state["messages"] + [
        ToolMessage(content = _Clip(result), tool_call_id = call["id"])
    ]
    return {
        "messages":   messages,
        "valid_ids":  state["valid_ids"] | _Ids_From_Result(call["name"], result),
        "tool_round": 2,
    }


def _Node_Final_Judge(state: RetrievalState) -> dict:
    """第三次 LLM 调用：只绑定 check_chunk 做最终判断。"""

    # 两次工具都用完了，这一轮只绑 check_chunk，模型要么直接作答要么触发兜底
    final = get_llm().bind_tools([CHECK_CHUNK_TOOL]).invoke(state["messages"])

    if final.tool_calls and any(tc["name"] == "check_chunk" for tc in final.tool_calls):
        return {"result_kind": "check"}

    text = _Guard_Citations(final.content, state["valid_ids"], "第三次最终判断")
    _Log_Agent_Result(text)
    return {"result_kind": "answer", "answer": text}


def _Node_Make_Partial(state: RetrievalState) -> dict:
    """check_chunk 触发后提取 partial，决定走 supplement 合并还是纯 chunk 兜底。"""

    # tool_round 是 1 还是 2，只影响日志文案，不影响判断逻辑本身
    first_round = state["tool_round"] == 1
    label   = "partial 摘要（首次工具后）" if first_round else "partial 摘要（第二次工具后）"
    partial = _Generate_Partial(state["messages"])

    # partial 非空且不是"无"，说明工具结果能确认一部分，走 supplement 跟 chunk 合并
    if partial and not partial.startswith("无"):
        partial = _Guard_Citations(partial, state["valid_ids"], label)
        if partial == _CITATION_REJECT:
            return {"result_kind": "answer", "answer": _CITATION_REJECT}
        if first_round:
            print("\n[Agent Result] 工具结果不完整，将与 chunk 合并回答。")
        else:
            print("\n[Agent Result] 两次工具结果不完整，将与 chunk 合并回答。")
        return {"result_kind": "supplement", "answer": partial}

    if first_round:
        print("\n[Agent Result] 工具无结果，回退至 chunk 兜底检索。")
    else:
        print("\n[Agent Result] 两次工具均无结果，回退至 chunk 兜底检索。")
    return {"result_kind": "fallback"}


def _Route_After_First(state: RetrievalState) -> str:
    # first_call 没调工具直接给了答案，result_kind 是 "answer"，子图直接结束；
    # 否则说明选了工具还没执行，走去 execute_tool 节点
    return "end" if state.get("result_kind") == "answer" else "execute"


def _Route_After_Judge(state: RetrievalState) -> str:
    # judge 节点写进 state 的三种 result_kind，分别对应三条不同的边
    kind = state["result_kind"]
    if kind == "answer":
        return "end"
    if kind == "check":
        return "partial"
    return "retry"


def _Route_After_Final(state: RetrievalState) -> str:
    # 第三次判断只可能是 check_chunk 兜底或者直接给答案，没有 retry 这条路了
    return "partial" if state["result_kind"] == "check" else "end"


def _Build_People_Graph():
    """组装并编译 people 检索子图。"""

    builder = StateGraph(RetrievalState)

    builder.add_node("first_call",    _Node_First_Call)
    builder.add_node("execute_tool",  _Node_Execute_Tool)
    builder.add_node("judge",         _Node_Judge)
    builder.add_node("execute_retry", _Node_Execute_Retry)
    builder.add_node("final_judge",   _Node_Final_Judge)
    builder.add_node("make_partial",  _Node_Make_Partial)

    builder.add_edge(START, "first_call")
    builder.add_conditional_edges(
        "first_call", _Route_After_First,
        {"execute": "execute_tool", "end": END},
    )
    builder.add_edge("execute_tool", "judge")
    builder.add_conditional_edges(
        "judge", _Route_After_Judge,
        {"retry": "execute_retry", "partial": "make_partial", "end": END},
    )
    builder.add_edge("execute_retry", "final_judge")
    builder.add_conditional_edges(
        "final_judge", _Route_After_Final,
        {"partial": "make_partial", "end": END},
    )
    builder.add_edge("make_partial", END)

    return builder.compile()


# 模块加载时编译子图，Route_People_Node 直接调用
_people_graph = _Build_People_Graph()


def Route_People_Node(task_text: str) -> str | None:
    """People 路由子图入口，保持原有返回契约不变。

    构造初始状态后调用编译好的子图，把终态 result_kind 映射回原契约：
        "answer"     → 文本答案字符串（含直接作答、最终判断、引用拒答）。
        "supplement" → "__SUPPLEMENT__" + partial，上游合并 chunk 一起回答。
        "fallback"   → None，由上游纯 chunk 兜底。

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
    final_state = _people_graph.invoke(init)

    kind   = final_state.get("result_kind", "")
    answer = final_state.get("answer", "")

    if kind == "supplement":
        return _SUPPLEMENT + answer
    if kind == "fallback":
        return None
    return answer
