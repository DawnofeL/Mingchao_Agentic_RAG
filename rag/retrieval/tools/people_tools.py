"""人物检索 LangChain tool 封装。

本模块把 people_store.py 提供的两个检索函数包装成 LangChain @tool，供 LLM 通过 native tool call 调用：
    people_search        — 粗粒度过滤查人物档案，接受 entities / era_filter / primary_filter。
    relationships_search — 按主体查关系图谱，接受 person / target / type_filter。
    check_chunk          — 信号工具，第二次 LLM 判断阶段使用，触发 chunk 兜底检索。

PEOPLE_TOOLS 列表由 route_people.py 注册给第一次 LLM 调用；CHECK_CHUNK_TOOL 注册给第二次判断调用。

注意：本模块里每个 @tool 函数的 docstring 不是写给人看的普通注释，@tool 会把它转成
JSON Schema 的 description 发给模型，模型推理时就是读这段文字来决定怎么填参数。
里面的枚举值、"填不准就传 null"之类的话都是说给模型听的，删减或精简前要确认不会
丢掉模型必须知道的约束（比如哪些参数只能取固定枚举值），不能当成普通代码注释随便瘦身。
"""

from langchain_core.tools import tool

from rag.retrieval.people_store import People_Search, Relationships_Search


@tool
def people_search(
    entities: list[str],
    era_filter: list[str] | None = None,
    primary_filter: str | None = None,
) -> list[dict]:
    """在人物库中按实体、时代、身份过滤，返回人物档案列表供 LLM 阅读理解。

    适用场景：不知道主体人名，需要通过别名、职衔、身份类别、时代找人；
    或已知人名但只需要人物档案信息（别名、官职、身份、时代、事件摘要）。

    设计原则：宁可多返回几条，让 LLM 从结果里读出答案，不靠参数精筛。

    Args:
        entities: 人名、别名、职衔关键词列表，无锚点时传空列表 []。
                  只能填人名 / 别名 / 职衔，严禁填事件描述、地点、形容词等非人名词语。
                  填不准宁可填 []，让 era_filter + primary_filter 宽泛召回。
        era_filter: 时代约束列表，列表内任意年号命中即保留（OR）；无时代约束时传 null。
                    只能取以下枚举值：至正 / 洪武 / 建文 / 永乐 / 洪熙 / 宣德 / 正统 /
                    景泰 / 天顺 / 成化 / 弘治 / 正德 / 嘉靖 / 隆庆 / 万历 /
                    泰昌 / 天启 / 崇祯 / 大义 / 天元 / 天历 / 天祐 / 太平。
                    跨年号时传列表：["洪武", "建文", "永乐"]。
        primary_filter: 身份大类约束，只能取以下枚举值之一，识别不到时传 null：
                        皇帝 / 明朝武将 / 文臣 / 宦官 / 皇室 / 反叛势力 /
                        势力 / 清 / 蒙古草原 / 朝鲜 / 日本 / 外国 / 社会人员。
    Returns:
        命中人物档案列表，每条含 name / aliases / primary_identity /
        secondary_identity / era / roles / events / summary。
    """

    return People_Search(
        entities       = entities,
        era_filter     = era_filter,
        primary_filter = primary_filter,
    )


@tool
def relationships_search(
    person: str,
    target: str | None = None,
) -> dict:
    """查询已知主体人物的关系图谱。

    适用场景：已知主体人名，查此人和谁有关联、与某人的具体关系、
    在某人麾下的职务、哪些人是其上级 / 下级 / 盟友 / 对手。
    返回全量关系列表，由 LLM 阅读判断。

    Args:
        person: 主体人名或别名，必填，在 name 和 aliases 字段做正则匹配。
        target: 关系目标人名，只返回 relationships[].target 匹配的条目；
                用于查"X 和 Y 什么关系"或"X 在 Y 麾下的职务"，无需时传 null。
    Returns:
        包含 name / era / relationships 的字典，relationships 是过滤后的关系列表，
        每条结构为 {type, target, context}。找不到主体时返回空字典。
    """

    return Relationships_Search(
        person = person,
        target = target,
    )


@tool
def check_chunk() -> str:
    """当 people 工具结果不能回答当前 task 时调用此工具，触发 chunk 兜底检索。

    使用场景：在第二次 LLM 判断阶段，如果工具返回的人物档案或关系图无法回答 task
    （结果为空、与 task 无关、或信息不足），调用此工具即可，不接受任何参数。
    调用此工具时严禁同时输出任何文字。
    """

    return ""


# 第一次 LLM 调用绑定的检索工具列表
PEOPLE_TOOLS = [people_search, relationships_search]

# 第二次 LLM 判断阶段绑定的信号工具
CHECK_CHUNK_TOOL = check_chunk
