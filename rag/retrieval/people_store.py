"""人物结构化检索模块。

本模块负责从 mingchao_people.json 加载人物数据，并提供两个检索函数：
    People_Search        — 三层过滤（era_filter → primary_filter → entities）依次收窄，返回人物档案。
    Relationships_Search — 按主体人名定位，再按 target 过滤关系条目，返回关系图。

两个函数供 people_tools.py 包装成 LangChain @tool，外部不直接调用本模块。
"""

import json
import re
from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "people_timeline" / "mingchao_people.json"

# 全局缓存，避免每次检索都重新读文件
_people_cache: list[dict] | None = None

# era_filter 合法枚举值，对应 people_tools.py 文档里允许 LLM 填写的年号表
_ERA_ENUM = {
    "至正", "洪武", "建文", "永乐", "洪熙", "宣德", "正统", "景泰", "天顺",
    "成化", "弘治", "正德", "嘉靖", "隆庆", "万历", "泰昌", "天启", "崇祯",
    "大义", "天元", "天历", "天祐", "太平",
}

# primary_filter 合法枚举值，对应 people_tools.py 文档里的 13 个身份大类
_PRIMARY_ENUM = {
    "皇帝", "明朝武将", "文臣", "宦官", "皇室", "反叛势力",
    "势力", "清", "蒙古草原", "朝鲜", "日本", "外国", "社会人员",
}


def _Validate_Era_Filter(era_filter: list[str] | None) -> list[str] | None:
    """剔除不在合法年号枚举表里的取值，打日志报警，避免非法值悄悄参与过滤。

    LLM 偶尔会填出表外的年号（拼错、用了清朝年号等），这类值不报错也不会
    被正则命中，过去只会安静地查不到东西，分不清是"真没有"还是"参数填错了"。

    Args:
        era_filter: LLM 填写的年号列表，可能为 None。
    Returns:
        剔除非法值后的列表；全部非法或原本为空时返回 None（跳过年代过滤）。
    """

    if not era_filter:
        return era_filter

    bad = [e for e in era_filter if e not in _ERA_ENUM]
    if bad:
        print(f"[参数校验报警] era_filter 出现非法枚举值 {bad}，已剔除，不参与过滤")

    good = [e for e in era_filter if e in _ERA_ENUM]
    return good or None


def _Validate_Primary_Filter(primary_filter: str | None) -> str | None:
    """校验 primary_filter 是否在合法身份大类枚举表里，不在表里就打日志报警并忽略。

    Args:
        primary_filter: LLM 填写的身份大类，可能为 None。
    Returns:
        合法则原样返回；非法返回 None（跳过身份过滤）。
    """

    if primary_filter is None:
        return None

    if primary_filter not in _PRIMARY_ENUM:
        print(f"[参数校验报警] primary_filter 出现非法枚举值 {primary_filter!r}，已忽略，不参与过滤")
        return None

    return primary_filter


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

    用的是 re.search，不要求从头匹配、也不要求整串匹配上，只要 pattern 在
    字符串里任意位置出现过一次就算命中，并且大小写不敏感。

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

    三个参数之间是 AND，依次收窄候选池（era_filter → primary_filter → entities）；
    每个参数自己内部如果是一组词，则是 OR，命中其中任意一个就算通过这一轮。
    entities 这一轮里，一个候选只要它的 name/aliases/roles/secondary_identity
    四个字段里，任意一个字段命中 entities 列表里任意一个词，就算通过。
    设计原则：宁可多返回，让 LLM 做阅读理解，而不是靠参数精筛导致漏召回。

    示例：entities=["朱元璋", "徐达"], era_filter=["洪武"], primary_filter="皇帝"。
    先留下 era 字段命中"洪武"的人物，再留下 primary_identity 正好是"皇帝"的，
    最后留下 name/aliases/roles/secondary_identity 任一字段命中"朱元璋"或"徐达"的，
    三轮过完，徐达因为不是"皇帝"会在第二轮被滤掉，朱元璋会留下来。

    Args:
        entities: 可能出现在 name/aliases/roles 字段的词语列表，无锚点时传空列表。
        era_filter: 时代枚举值列表，任意枚举命中即保留（OR），为 None 时跳过时代过滤。
        primary_filter: 身份大类枚举值，精确匹配 primary_identity，为 None 时跳过。
    Returns:
        命中人物档案列表，每条含 name / aliases / primary_identity /
        secondary_identity / era / roles / events / summary，不含 relationships 和 source_chunks。
    """

    era_filter     = _Validate_Era_Filter(era_filter)
    primary_filter = _Validate_Primary_Filter(primary_filter)

    candidates = _Load_Store()

    # 第一轮：era_filter OR 约束，任意年号命中即保留
    if era_filter:
        candidates = [
            p for p in candidates
            if any(_Regex_Match_Field(p.get("era", ""), era) for era in era_filter)
        ]

    # primary_identity 精确等于 primary_filter 才算命中
    def _primary_hit(p: dict) -> bool:
        return p.get("primary_identity", "") == primary_filter

    # entities 里任意一个词，在 name/aliases/roles/secondary_identity
    # 任意一个字段命中，就算这个候选通过
    def _entities_hit(p: dict) -> bool:
        return any(
            _Regex_Match_Field(p.get("name", ""), kw)
            or _Regex_Match_Field(p.get("aliases", []), kw)
            or _Regex_Match_Field(p.get("roles", []), kw)
            or _Regex_Match_Field(p.get("secondary_identity", []), kw)
            for kw in entities
        )

    # 第二轮：用 primary_filter 收窄
    if primary_filter:
        candidates = [p for p in candidates if _primary_hit(p)]

    # 第三轮：用 entities 继续收窄
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

    先在 name 和 aliases 字段定位主体人物（OR 匹配，取第一个命中），再用 target
    过滤该主体的 relationships 列表，只看 relationships[].target 这个字段
    （对方是谁），不看关系类型。target 为 None 时返回全量关系。

    示例：person="朱元璋", target="马皇后"，假设朱元璋的 relationships 里有
    [{"target": "马皇后", "relation": "皇后"}, {"target": "朱标", "relation": "长子"}]，
    过滤后只剩 target 匹配"马皇后"的那一条，"朱标"这条会被丢掉。

    Args:
        person: 主体人名或别名，必填，在 name 和 aliases 字段做正则匹配，取第一个命中的记录。
        target: 关系目标人名，只保留 relationships[].target 匹配的条目，为 None 时不过滤。
    Returns:
        包含 name / era / relationships 的字典，relationships 是过滤后的关系列表。
        找不到主体时返回空字典。
    """

    store = _Load_Store()

    # 定位主体：逐条记录检查 person 是否匹配 name 或 aliases，取第一个命中的
    matched = None
    for p in store:
        name_hit = _Regex_Match_Field(p.get("name", ""), person)
        alias_hit = _Regex_Match_Field(p.get("aliases", []), person)

        if name_hit or alias_hit:
            matched = p
            break

    if matched is None:
        return {}

    # 过滤关系：只留 target 字段（关系对象是谁）匹配的条目，关系类型不参与过滤
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
