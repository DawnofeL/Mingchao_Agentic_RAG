"""人物结构化检索模块。

本模块负责从 mingchao_people.json 加载人物数据，并提供两个检索函数：
    People_Search        — 三层过滤（era_filter → primary_filter → entities）依次收窄，返回人物档案。
    Relationships_Search — 按主体人名定位，再按 target / type_filter 过滤关系条目，返回关系图。

两个函数供 people_tools.py 包装成 LangChain @tool，外部不直接调用本模块。
"""

import json
import re
from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "people_timeline" / "mingchao_people.json"

# 全局缓存，避免每次检索都重新读文件
_people_cache: list[dict] | None = None


def _Load_Store() -> list[dict]:
    """加载人物数据，首次调用后缓存到模块全局变量，后续直接返回缓存。

    Returns:
        人物记录列表，每条是一个包含完整字段的字典。
    """

    global _people_cache

    if _people_cache is not None:
        return _people_cache

    with open(_DATA_PATH, encoding = "utf-8") as f:
        _people_cache = json.load(f)

    return _people_cache


def _Regex_Match_Field(field_value: str | list, pattern: str) -> bool:
    """对字符串或字符串列表字段做正则子串匹配，任一命中返回 True。

    Args:
        field_value: 目标字段值，可以是字符串或字符串列表。
        pattern: 正则表达式字符串。
    Returns:
        任一元素命中则返回 True，否则 False。
    """

    if isinstance(field_value, list):
        return any(re.search(pattern, str(v), re.IGNORECASE) for v in field_value)

    return bool(re.search(pattern, str(field_value), re.IGNORECASE))


def People_Search(
    entities: list[str],
    era_filter: list[str] | None,
    primary_filter: str | None,
) -> list[dict]:
    """三参数 AND 收窄检索人物档案。

    参数间 AND（era → primary_filter → entities 依次收窄），参数内 list 是 OR。
    设计原则：宁可多返回，让 LLM 做阅读理解，而不是靠参数精筛导致漏召回。

    Args:
        entities: 可能出现在 name/aliases/roles 字段的词语列表，无锚点时传空列表。
        era_filter: 时代枚举值列表，任意枚举命中即保留（OR），为 None 时跳过时代过滤。
        primary_filter: 身份大类枚举值，精确匹配 primary_identity，为 None 时跳过。
    Returns:
        命中人物档案列表，每条含 name / aliases / primary_identity /
        secondary_identity / era / roles / events / summary，不含 relationships 和 source_chunks。
    """

    candidates = _Load_Store()

    # 第一轮：era_filter OR 约束，任意年号命中即保留
    if era_filter:
        candidates = [
            p for p in candidates
            if any(_Regex_Match_Field(p.get("era", ""), era) for era in era_filter)
        ]

    # 第二轮：primary_filter AND entities，依次收窄
    def _primary_hit(p: dict) -> bool:
        return p.get("primary_identity", "") == primary_filter

    def _entities_hit(p: dict) -> bool:
        return any(
            _Regex_Match_Field(p.get("name", ""), kw)
            or _Regex_Match_Field(p.get("aliases", []), kw)
            or _Regex_Match_Field(p.get("roles", []), kw)
            or _Regex_Match_Field(p.get("secondary_identity", []), kw)
            for kw in entities
        )

    if primary_filter:
        candidates = [p for p in candidates if _primary_hit(p)]
    if entities:
        candidates = [p for p in candidates if _entities_hit(p)]

    # 只保留需要的字段，去掉 relationships 和 source_chunks
    keep_fields = {"people_id", "name", "aliases", "primary_identity", "secondary_identity", "era", "roles", "events", "summary"}

    return [{k: v for k, v in p.items() if k in keep_fields} for p in candidates]


def Relationships_Search(
    person: str,
    target: str | None,
) -> dict:
    """检索主体人物的关系图谱。

    先在 name 和 aliases 字段定位主体人物，再按 target 过滤 relationships 列表。
    target 为 None 时返回全量关系，让 LLM 做阅读理解。

    Args:
        person: 主体人名或别名，必填，在 name 和 aliases 字段做正则匹配。
        target: 关系目标人名，只保留 relationships[].target 匹配的条目，为 None 时不过滤。
    Returns:
        包含 name / era / relationships 的字典，relationships 是过滤后的关系列表。
        找不到主体时返回空字典。
    """

    store = _Load_Store()

    # 在 name 和 aliases 字段定位主体人物，取第一个命中的记录
    matched = None
    for p in store:
        name_hit = _Regex_Match_Field(p.get("name", ""), person)
        alias_hit = _Regex_Match_Field(p.get("aliases", []), person)

        if name_hit or alias_hit:
            matched = p
            break

    if matched is None:
        return {}

    # 取出关系列表，target 过滤：只保留 relationships[].target 匹配的条目
    relationships = matched.get("relationships", [])

    if target is not None:
        relationships = [
            r for r in relationships
            if _Regex_Match_Field(r.get("target", ""), target)
        ]

    return {
        "people_id":     matched.get("people_id"),
        "name":          matched.get("name", ""),
        "era":           matched.get("era", ""),
        "relationships": relationships,
    }
