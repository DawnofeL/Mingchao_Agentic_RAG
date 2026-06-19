"""时序事件检索 LangChain tool 封装。

本模块把 timeline_store.py 提供的检索函数包装成 LangChain @tool，供 LLM 通过 native tool call 调用：
    event_search — 四参数过滤事件列表：event_keywords / era / year / participants。

TIMELINE_TOOLS 由 route_timeline.py 注册给第一次 LLM 调用；
CHECK_CHUNK_TOOL 复用 people_tools.py 的定义，由第二次判断调用绑定。
"""

from langchain_core.tools import tool

from rag.retrieval.timeline_store import Event_Search


@tool
def event_search(
    event_keywords: list[str] | None = None,
    era: list[str] | None = None,
    year: list[int] | None = None,
    participants: list[str] | None = None,
) -> list[dict]:
    """在时序事件库中检索事件，返回事件列表供 LLM 阅读理解。

    参数之间是 AND（全部非空条件同时生效），list 参数内部是 OR（任意元素命中即保留）。

    Args:
        event_keywords: 事件名、历史术语、作为核心锚点的地名，打 event/tags/location/outcome 字段。
                        严禁填人名（→ participants）、年号年份（→ era/year）、泛化动词。
                        无事件锚点时传 []。
        era: 年号列表，列表内任意年号命中即保留（OR）；无时代约束时传 null。
             只能取以下枚举值：天历 / 至正 / 洪武 / 建文 / 永乐 / 洪熙 / 宣德 / 正统 /
             景泰 / 天顺 / 成化 / 弘治 / 正德 / 嘉靖 / 隆庆 / 万历 / 泰昌 / 天启 / 崇祯。
             单个年号：["永乐"]；跨年号：["建文", "永乐"]；识别不到传 null。
        year: 公元年整数列表，任意年份精确命中即保留（OR）；无时传 null。
              仅当 task 明确出现公元年数字时填，中文年号一律走 era，严禁自行换算。
              单年：[1402]；多年：[1399, 1402]。
        participants: 人名、别名、称号列表，打 participants 字段。
                      严禁填物品、地点、事件名、朝代名。无人物锚点传 []。
    Returns:
        命中事件列表，每条含 event_id / year / era / event / tags /
        location / participants / outcome / summary。
    """

    return Event_Search(
        event_keywords = event_keywords,
        era            = era,
        year           = year,
        participants   = participants,
    )


TIMELINE_TOOLS = [event_search]
