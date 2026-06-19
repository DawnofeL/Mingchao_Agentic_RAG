"""单任务路由节点。

本模块根据 QU 输出的 plan 做两层路由，并提供 people / timeline / direct 分支的执行入口：
    Route_Task          — 两层路由入口，subtasks 占位，single 按 intention 分流。
    Synthesize_Answer   — 接收 query + chunks（+ 可选 structured_context），调 LLM 综合回答。
    _Run_People         — people 双路并行节点：副线程始终跑 chunk 检索；
                          主线程并行跑 Route_People_Node：
                            完整答案 → 直接返回；
                            "__SUPPLEMENT__" + partial → 合并 chunk 一起合成；
                            None → 纯 chunk 兜底。
    _Run_Timeline       — timeline / chunk 双路并行节点，结构与 _Run_People 完全对称。
    _Run_Direct         — direct 分支：调 LLM 基于多轮对话历史直接回复。
"""

from concurrent.futures import ThreadPoolExecutor

import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag.config.settings import get_llm
from rag.graph.citation_check import Validate_Citations
from rag.graph.nodes.orchestrator import Run_Orchestrator
from rag.graph.nodes.route_people import Route_People_Node, _SUPPLEMENT as _PEOPLE_SUPPLEMENT
from rag.graph.nodes.route_timeline import Route_Timeline_Node, _SUPPLEMENT as _TIMELINE_SUPPLEMENT
from rag.retrieval.chunk_rrf import Search_Chunks


# fact 类问题：严格逐句引用，每条陈述必须有原文直接支撑
_SYNTHESIZE_FACT_PROMPT = """\
你是一个历史问答助手。根据下方提供的原文段落回答用户的问题。

你是一个只能看到以下段落的失忆读者，对这段历史没有任何先验知识。无论你认为某个细节在历史上是否正确，只要它没有出现在下方段落中，就严禁写出。

规则：
- 只能使用下方段落中明确出现的信息作答，严禁引入段落以外的任何内容——即使你认为那是正确的历史知识。
- 每一条事实/陈述句末必须跟 [chunk_id=N]，N 为段落标题中显示的 chunk_id。
- 同一句话中同一个 chunk_id 只标一次，出现在句末；不在句中重复出现。
- 同一条事实有多个段落支持时，写多个 id：[chunk_id=N,M]。
- 没有 chunk_id 锚定的内容严禁写出。
- 能回答：直接作答，task 问什么就答什么，严禁附加任何 task 未要求的背景、解释、生平、评价。
- 找不到答案：只输出"根据现有资料，无法回答此问题。"，不做任何补充。
- 两种情况都必须输出文本，严禁静默。

**答案铁律：答案必须是能完整回答 task 的最短自然语言语句，严禁任何展开。**

✅ task: "姚广孝怎么暗示朱棣称帝的？" → 姚广孝向朱棣赠送白帽来暗示其称帝[chunk_id=189]。（只描述行为本身，不加背景铺垫，没有任何多余展开）
❌ task: "姚广孝怎么暗示朱棣称帝的？" → 姚广孝是朱棣的头号谋臣，靖难之役中出谋划策。他以赠送白帽的方式……—— task 只问暗示方式，冒号前全是废话。

反例（严禁出现）：
❌ 朱棣登基后年号永乐，迁都北京。[chunk_id=10] —— 原文只说"登基"，"年号永乐""迁都北京"若未在段落中出现，不得补充，哪怕你知道这是事实。
❌ 方孝孺被灭十族，株连甚广。—— 未标 chunk_id，严禁写出。\
"""


# analysis / multi_enum 类问题：允许基于多条 chunk 证据推断结论，但仍须标注支撑 chunk
_SYNTHESIZE_ANALYSIS_PROMPT = """\
你是一个历史问答助手。根据下方提供的原文段落，综合分析并回答用户的问题。

你是一个只能看到以下段落的失忆读者，对这段历史没有任何先验知识。无论你认为某个细节在历史上是否正确，只要它没有出现在下方段落中，就严禁写出。

此类问题需要从多条段落中综合推断（如"谁的功劳最大"、"各自结果如何"等），原文不会直接写出结论，但可以基于段落证据进行有限推断。

规则：
- 只能使用下方段落中的信息作答，严禁引入段落以外的任何内容——即使你认为那是正确的历史知识。
- 推断性结论必须标注所有支撑它的 chunk_id：[chunk_id=N,M,...]。
- 直接转述的事实，句末跟 [chunk_id=N]。
- 没有任何 chunk 支撑的内容严禁写出。
- 推断要保守：段落证据明确指向某结论时才写出；证据模糊时说"段落显示……但未明确比较……"。
- 找不到任何相关内容：只输出"根据现有资料，无法回答此问题。"
- 必须输出文本，严禁静默。

**答案铁律：答案必须是能完整回答 task 的最短自然语言语句，task 问什么就答什么，严禁附加任何 task 未要求的背景、解释、生平、评价。**

✅ task: "姚广孝怎么暗示朱棣称帝的？" → 姚广孝向朱棣赠送白帽来暗示其称帝[chunk_id=189]。（只描述行为本身，不加背景铺垫，没有任何多余展开）
❌ task: "姚广孝怎么暗示朱棣称帝的？" → 姚广孝是朱棣的头号谋臣，靖难之役中出谋划策。他以赠送白帽的方式……—— task 只问暗示方式，冒号前全是废话。

反例（严禁出现）：
❌ 朱能是靖难中功劳最大的武将，因为他作战勇猛。—— 无 chunk_id 支撑，不得写出。
❌ 综合来看，朱能功劳最大。[chunk_id=239] —— 若 chunk_id=239 只说"朱能救了朱棣"，而未评价其为功劳最大，不得直接得出此结论。
✅ 段落显示朱能多次在危急时刻救援朱棣 [chunk_id=239,245]，是靖难中持续发挥关键作用的武将；但段落未直接比较诸将功劳大小，无法确定谁为"最大"。\
"""


_SYNTHESIZE_PROMPTS = {
    "fact":       _SYNTHESIZE_FACT_PROMPT,
    "analysis":   _SYNTHESIZE_ANALYSIS_PROMPT,
    "multi_enum": _SYNTHESIZE_ANALYSIS_PROMPT,
}


# Direct 回复 system prompt：基于历史对话直接回复，并保留前文已有的 id 锚点
_DIRECT_PROMPT = """\
你是一个历史问答助手，正在与用户进行多轮对话。当前消息不需要触发检索，请直接回复。

规则：
- 如果用户在闲聊、感叹、或请求你做某件事（例如重复、总结、换个说法），正常回应即可，不要拒绝或说无法回答。
- 如果用户在追问历史事实，且历史对话中有相关证据，**保留并复用历史对话中已有的 [people_id=N] / [chunk_id=N] 锚点**，不要丢弃。
- 如果用户在追问历史事实，但历史对话中确实没有相关证据，才回复不知道，严禁静默。
- 严禁编造新的 id，只能复述历史对话中实际出现过的 id。
- 简洁直接，不拓展延伸，不做无关解释。\
"""


def Route_Task(plan: dict, history: list = [], top_k: int = 10) -> str:
    """根据 QU 输出的 plan 路由到对应执行分支，返回最终答案字符串。

    Args:
        plan:    Query_Understanding_Node 返回的查询计划字典，
                 含 refined_query / task_type / tasks。
        history: 多轮对话历史，每条含 role / content；direct 分支需要用到。
    """

    if plan["task_type"] == "subtasks":
        print("[Router] 多任务 → 路由至 Orchestrator")
        return Run_Orchestrator(plan, history, top_k=top_k)

    task      = plan["tasks"][0]
    intention = task["intention"]
    task_text = task["task"]

    print(f"[Router] 单任务 · intention={intention}")
    print()

    query_kind = task.get("query_kind", "fact")

    if intention == "people":
        return _Run_People(task_text, query_kind=query_kind, top_k=top_k)

    elif intention == "timeline":
        return _Run_Timeline(task_text, query_kind=query_kind, top_k=top_k)

    elif intention == "direct":
        return _Run_Direct(task_text, history)

    else:
        raise ValueError(f"未知 intention: {intention!r}")


def Synthesize_Answer(
    query: str,
    chunks: list[dict],
    query_kind: str = "fact",
    structured_context: str | None = None,
) -> str:
    """用 LLM 综合 chunk 内容（及可选的结构化数据）回答用户问题。

    根据 query_kind 选择 prompt：fact 严格逐句引用；analysis/multi_enum 允许推断。
    生成后校验答案里的 [chunk_id=N] 是否全部来自传入的 chunks，校验不过打日志报警并拒答。

    Args:
        query:              用户原始问题。
        chunks:             Search_Chunks 返回的 chunk 列表，每条含 chunk_id / content 字段。
        query_kind:         QU 输出的查询类型，"fact" / "analysis" / "multi_enum"。
        structured_context: people / timeline 路由提取的 partial answer（带 id 引用），
                            与 chunk 段落合并后一起送给 LLM 综合。为 None 时只用 chunk。
    Returns:
        LLM 综合后的答案字符串。无任何内容时直接返回"找不到"提示。
    """

    if not chunks and not structured_context:
        return "根据现有资料，无法回答此问题。"

    context_parts = []
    if structured_context:
        context_parts.append(f"【结构化数据已确认部分（带 id 引用，可直接引用）】\n{structured_context}")
    for chunk in chunks:
        cid = chunk.get("chunk_id", "?")
        context_parts.append(f"[chunk_id={cid}]\n{chunk.get('content', '')}")
    context_text = "\n\n".join(context_parts)

    if structured_context:
        human_text = (
            f"问题：{query}\n\n"
            f"参考内容（【结构化数据】和原文段落均为有效来源，均可直接引用）：\n{context_text}"
        )
    else:
        human_text = f"问题：{query}\n\n参考段落：\n{context_text}"

    prompt = _SYNTHESIZE_PROMPTS.get(query_kind, _SYNTHESIZE_FACT_PROMPT)
    messages = [
        SystemMessage(content = prompt),
        HumanMessage(content  = human_text),
    ]

    try:
        response = get_llm().invoke(messages)
    except openai.BadRequestError as e:
        print(f"[Synthesize] API 内容审核拦截，跳过本次合成：{e}")
        return "（API 内容审核拦截，无法生成回答）"

    answer = response.content.strip() if response.content else ""
    if not answer:
        return "（LLM 返回了空响应，无法给出结论）"

    valid_chunk_ids = {c["chunk_id"] for c in chunks if isinstance(c.get("chunk_id"), int)}
    error = Validate_Citations(answer, "chunk_id", valid_chunk_ids)
    if error is not None:
        print(f"\n[Citation 报警] Synthesize_Answer：{error}")
        return "根据现有资料，无法回答此问题。"

    return answer


def _Print_Chunk_Summary(chunks: list[dict], header: str) -> None:
    """打印 chunk 检索摘要：每条 chunk_id / volume / index / rerank / rrf + 前 100 字。"""

    if not chunks:
        print(f"[{header}] 未检索到相关内容\n")
        return

    print(f"[{header}] 检索到 {len(chunks)} 条结果\n")
    for i, chunk in enumerate(chunks, 1):
        meta = (
            f"[{i}] chunk_id={chunk.get('chunk_id', '?')}  "
            f"vol={chunk.get('volume', '?')}  "
            f"chunk={chunk.get('chunk_index', '?')}/{chunk.get('chunk_total', '?')}  "
            f"rerank={chunk.get('reranker_score', 0.0):.4f}  "
            f"rrf={chunk.get('rrf_score', 0.0):.6f}"
        )
        preview = chunk.get("content", "")[:100]
        print(meta)
        print(preview)
        print()


def _Run_People(task_text: str, query_kind: str = "fact", top_k: int = 10) -> str:
    """People / chunk 双路并行节点。

    副线程始终启动 chunk 检索。主线程并行跑 Route_People_Node：
      - 完整答案 → 直接返回，不等 chunk。
      - "__SUPPLEMENT__" + partial → 等 chunk 结果，合并后一起送 LLM 综合。
      - None → 纯 chunk 兜底合成。
    """

    with ThreadPoolExecutor(max_workers = 1) as executor:
        chunk_thread = executor.submit(Search_Chunks, task_text, top_k)

        people_answer = Route_People_Node(task_text)

        # 完整答案，不需要 chunk
        if people_answer is not None and not people_answer.startswith(_PEOPLE_SUPPLEMENT):
            return people_answer

        chunks = chunk_thread.result()
        print()
        _Print_Chunk_Summary(chunks, header = "Chunk · 兜底")

        structured = None
        if people_answer is not None and people_answer.startswith(_PEOPLE_SUPPLEMENT):
            structured = people_answer[len(_PEOPLE_SUPPLEMENT):]

        label  = "People + Chunk 合并" if structured else "Chunk · 兜底"
        answer = Synthesize_Answer(task_text, chunks, query_kind=query_kind, structured_context=structured)
        print(f"[Synth · {label}]")   # 只留路径标记，答案走返回值不再打印
        return answer


def _Run_Timeline(task_text: str, query_kind: str = "fact", top_k: int = 10) -> str:
    """Timeline / chunk 双路并行节点，结构与 _Run_People 完全对称。

    副线程始终启动 chunk 检索。主线程跑 Route_Timeline_Node：
      - 完整答案 → 直接返回，不等 chunk。
      - "__SUPPLEMENT__" + partial → 等 chunk 结果，合并后一起送 LLM 综合。
      - None → 纯 chunk 兜底合成。
    """

    with ThreadPoolExecutor(max_workers = 1) as executor:
        chunk_thread = executor.submit(Search_Chunks, task_text, top_k)

        timeline_answer = Route_Timeline_Node(task_text)

        # 完整答案，不需要 chunk
        if timeline_answer is not None and not timeline_answer.startswith(_TIMELINE_SUPPLEMENT):
            return timeline_answer

        chunks = chunk_thread.result()
        print()
        _Print_Chunk_Summary(chunks, header = "Chunk · 兜底")

        structured = None
        if timeline_answer is not None and timeline_answer.startswith(_TIMELINE_SUPPLEMENT):
            structured = timeline_answer[len(_TIMELINE_SUPPLEMENT):]

        label  = "Timeline + Chunk 合并" if structured else "Chunk · 兜底"
        answer = Synthesize_Answer(task_text, chunks, query_kind=query_kind, structured_context=structured)
        print(f"[Synth · {label}]")   # 只留路径标记，答案走返回值不再打印
        return answer


def _Run_Direct(task_text: str, history: list) -> str:
    """Direct 分支：基于多轮对话历史直接回复，无检索。

    把 history 还原成 LLM 的对话消息序列，配合 _DIRECT_PROMPT 让 LLM
    基于前文回复。前文回答里若有 [people_id=N] / [chunk_id=N] 锚点，
    LLM 必须保留并复用，确保追问真伪时能诚实自审。

    Args:
        task_text: 当前用户问题。
        history:   多轮对话历史列表，每条含 role ("user" | "assistant") / content。
    """

    messages = [SystemMessage(content = _DIRECT_PROMPT)]

    # 把历史对话还原成 LLM message 序列
    for turn in history:
        role    = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content = content))
        elif role == "assistant":
            messages.append(AIMessage(content = content))

    messages.append(HumanMessage(content = task_text))

    response = get_llm().invoke(messages)

    answer = response.content.strip() if response.content else ""
    if not answer:
        answer = "（LLM 返回了空响应，无法给出结论）"

    return answer
