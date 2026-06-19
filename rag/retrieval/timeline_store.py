"""时序事件结构化检索模块。

本模块负责从 mingchao_timeline.json 加载事件数据，并提供检索函数：
    Event_Search — 四参数过滤（event_keywords / era / year / participants），返回事件列表。

供 timeline_tools.py 包装成 LangChain @tool，外部不直接调用本模块。
"""

import json
import re
from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "people_timeline" / "mingchao_timeline.json"

_timeline_cache: list[dict] | None = None

# 返回时保留的字段（source_chunks 不对外暴露）
_KEEP_FIELDS = {"event_id", "year", "era", "event", "tags", "location", "participants", "outcome", "summary"}

# era 合法枚举值
_ERA_ENUM = {
    "天历", "至正",
    "洪武", "建文", "永乐", "洪熙", "宣德",
    "正统", "景泰", "天顺", "成化", "弘治",
    "正德", "嘉靖", "隆庆", "万历", "泰昌",
    "天启", "崇祯",
}


def _Load_Store() -> list[dict]:
    global _timeline_cache
    if _timeline_cache is not None:
        return _timeline_cache
    with open(_DATA_PATH, encoding="utf-8") as f:
        _timeline_cache = json.load(f)
    return _timeline_cache


def _Regex_Match(value: str | list, pattern: str) -> bool:
    """对字符串或字符串列表做正则子串匹配，任一命中返回 True。"""
    if isinstance(value, list):
        return any(re.search(pattern, str(v), re.IGNORECASE) for v in value)
    return bool(re.search(pattern, str(value), re.IGNORECASE))


def Event_Search(
    event_keywords: list[str] | None,
    era: list[str] | None,
    year: list[int] | None,
    participants: list[str] | None,
) -> list[dict]:
    """四参数过滤检索事件列表。

    参数间 AND（全部非 null 条件同时生效），参数内 list 是 OR（任意元素命中即保留）。
    event_keywords 打 event / tags / location / outcome 四个字段；
    era 列表内任意年号子串命中即保留；year 列表内任意公元年精确命中即保留；
    participants 打 participants 字段。

    Args:
        event_keywords: 事件名、历史术语、地名关键词列表，null 时跳过。
        era: 年号枚举值列表，任意年号命中即保留（OR）；非法枚举值自动忽略；null 时跳过。
        year: 公元年整数列表，任意年份精确命中即保留（OR）；null 时跳过。
        participants: 人名、别名、称号列表，null 时跳过。
    Returns:
        命中事件列表，每条含 event_id / year / era / event / tags /
        location / participants / outcome / summary。
    """

    candidates = _Load_Store()

    # era 过滤：过滤掉非法枚举值，剩余年号 OR 匹配 era 字段
    if era is not None:
        valid_eras = [e for e in era if e in _ERA_ENUM]
        if not valid_eras:
            return []
        candidates = [
            e for e in candidates
            if any(_Regex_Match(e.get("era", ""), era_val) for era_val in valid_eras)
        ]

    # year 过滤：任意年份精确命中即保留（OR）
    if year is not None:
        candidates = [e for e in candidates if e.get("year") in year]

    # participants 过滤：OR，任意人名命中即保留
    if participants:
        candidates = [
            e for e in candidates
            if any(_Regex_Match(e.get("participants", []), name) for name in participants)
        ]

    # event_keywords 过滤：OR，任意词在 event/tags/location/outcome/summary 任意字段命中即保留
    if event_keywords:
        target_fields = ["event", "tags", "location", "outcome", "summary"]
        candidates = [
            e for e in candidates
            if any(
                _Regex_Match(e.get(field, ""), kw)
                for kw in event_keywords
                for field in target_fields
            )
        ]

    return [{k: v for k, v in e.items() if k in _KEEP_FIELDS} for e in candidates]
