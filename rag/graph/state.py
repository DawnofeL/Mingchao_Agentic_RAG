"""LangGraph 状态定义。

本模块定义 Vector Mode 和 Agentic Mode 用到的所有状态 TypedDict（LangGraph 里
StateGraph 的状态容器类型），以及配合 OrchestratorState.pool 字段使用的合并函数。
各状态按图的嵌套层级从外到内排列：

    VectorState         Vector 模式的状态，全图只有一个检索节点。
    AgenticState         Agentic 模式顶层图的状态，按问题路由到下面两类分支之一。
    merge_pool           Orchestrator 结果池专用的合并函数，传给 OrchestratorState.pool
                          的 Annotated 标注使用。
    OrchestratorState     多任务编排子图的状态，由顶层图的编排分支调用。
    RetrievalState        people / timeline 检索子图的状态，由顶层图单任务分支
                          和编排子图的 worker 共同调用。
"""

from typing import Annotated, TypedDict


class VectorState(TypedDict):

    # 用户输入问题
    raw_query: str

    # Search_Chunks 返回的 chunk 列表，图执行结束后读取此字段
    chunks: list

    # 最终返回 chunk 数，传入 Search_Chunks
    top_k: int


class AgenticState(TypedDict):
    """Agentic Mode 顶层图的状态。

    顶层图很浅：qu 节点出 plan，再按 task_type / intention 路由到单任务分支或编排分支，
    各分支把答案写进 final_answer。编排和检索都作为子图在分支节点内部调用。
    """

    # 用户原始问题（输入）
    raw_query: str

    # 多轮对话历史
    history: list

    # 检索 top_k
    top_k: int

    # Query_Understanding_Node 输出的查询计划
    plan: dict

    # 最终答案（输出）
    final_answer: str


def merge_pool(left: dict, right: dict) -> dict:
    """Orchestrator 结果池的合并函数。

    多个 worker 在同一 superstep 并行写结果池，LangGraph 默认行为是
    谁写得晚就整个覆盖掉 pool 字段，先写的结果会丢。所以这个函数被
    传给 OrchestratorState.pool 的 Annotated 标注（见下方），
    告诉 LangGraph 改用这个函数的逻辑去合并，而不是直接覆盖：
    按 task_id 把各 worker 返回的结果列表拼起来，同一 task_id 下追加，
    新 task_id 直接新建。orchestrator 写阻塞结果走的也是这套合并逻辑。

    例：
        left  = {"t1": ["A"], "t2": ["空印案结果"]}
        right = {"t2": ["胡惟庸案结果"]}
        合并后 = {"t1": ["A"], "t2": ["空印案结果", "胡惟庸案结果"]}

    Args:
        left: 当前已累积的结果池。
        right: 本次节点返回需要合并进来的结果池。
    Returns:
        合并后的新结果池，不修改入参。
    """

    # 先把已有结果池里每个 task_id 对应的列表浅拷贝一份，避免直接改 left
    merged = {k: list(v) for k, v in left.items()}

    # 新结果逐个 task_id 拼接进去，已存在的追加，不存在的新建空列表再追加
    for task_id, results in right.items():
        merged.setdefault(task_id, [])
        merged[task_id] = merged[task_id] + list(results)

    return merged


class OrchestratorState(TypedDict):
    """多子任务编排图的状态。

    orchestrator 节点与 worker 节点交替推进：orchestrator 算出本轮 ready 任务并扇出，
    worker 并行执行后把结果合并回 pool，再回到 orchestrator 算下一轮。
    pool 是唯一会被多个 worker 并行写入的字段，所以挂 merge_pool reducer，其余字段
    都只由 orchestrator 单独写，用默认覆盖即可。
    """

    # QU 输出的精化问题，贯穿全程
    refined_query: str

    # 多轮对话历史，供 direct 子任务使用
    history: list

    # 检索 top_k，透传给各 worker
    top_k: int

    # 还没派发的子任务清单
    pending: list

    # 结果池，task_id → TaskResult 列表，Annotated 标注让 LangGraph 用 merge_pool 合并
    pool: Annotated[dict, merge_pool]

    # 本轮算出的待执行 job，供路由扇出
    jobs: list

    # 当前轮次，仅用于日志
    round_num: int

    # orchestrator 给路由的指示："dispatch" / "loop" / "synthesize"
    route: str

    # 终答合成结果（输出）
    final_answer: str


class RetrievalState(TypedDict):
    """people / timeline 检索子图共享的状态。

    子图内部是线性推进，没有并行分支，所以各字段直接覆盖写入即可，
    不需要像 OrchestratorState.pool 那样配专门的合并函数。

    拿 task_text="朱棣的首席谋士是谁？" 走一遍这套字段是怎么被填的：
        1. LLM 决定先查 people_search，这个决定写进 pending_tool，
           tool_round 标 1，表示这是第一次尝试的工具。
        2. 工具真的跑完查到了姚广孝的档案，这条记录追加进 messages，
           记录里带的 people_id=12 这个编号同时存进 valid_ids，
           以后写答案如果引用了不在这个集合里的编号，就会被判定
           是编出来的，不让它流出去。
        3. 这次查到的内容够回答问题了，所以 result_kind 标成
           "answer"，answer 写成"朱棣的首席谋士是姚广孝[people_id=12]"，
           子图到这里就结束了。
        4. 如果第 2 步没查到东西，不会在 people_search 上反复改参数重试，
           而是换 relationships_search 再试一次，tool_round 变成 2。
        5. 两个工具都试过还是没查到，就放弃这条结构化检索，转去原文
           chunk 搜索兜底，result_kind 标成 "fallback"。
        6. 如果查到了但不全（比如问题要列五个人只查到三个），就标成
           "supplement"，介于答上了和答不上之间。
    """

    # 当前子任务问题文本（输入）
    task_text: str

    # 子图内累积的对话消息序列
    messages: list

    # 下一步要执行的工具，含 name / args / id
    pending_tool: dict

    # 已执行工具贡献的合法引用 id，供答案锚点校验
    valid_ids: set

    # 已执行到第几个工具（1 / 2），决定 partial 日志措辞
    tool_round: int

    # 终态分类："answer" / "supplement" / "fallback"
    result_kind: str

    # 文本答案或 partial 摘要（输出）
    answer: str
