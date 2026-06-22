"""引用锚点校验。

校验 LLM 生成的答案文本里 [people_id=N] / [event_id=N] / [chunk_id=N] 这类引用锚点，
是不是都能在本轮真实证据池里找到对应的 id，防止 LLM 编造或挪用不存在的引用。
本模块定义：
    Extract_Cited_Ids   — 从答案文本里抠出指定字段的所有引用 id。
    Validate_Citations  — 调用 Extract_Cited_Ids，再跟合法 id 集合比对差集。
本模块只做"抠 id + 比对"，校验失败之后怎么处理（打日志、拒答）交给各调用方决定。
"""

import re

_CITATION_PATTERN = re.compile(r"\[(people_id|event_id|chunk_id)=([\d,\s]+)\]")


def Extract_Cited_Ids(text: str, field: str) -> set[int]:
    """从答案文本里抠出指定字段的所有引用 id，兼容逗号并列写法（如 [chunk_id=10,12]）。

    Args:
        text: 待检查的答案文本。
        field: 引用字段名，"people_id" / "event_id" / "chunk_id"。
    Returns:
        答案里实际引用到的 id 集合。
    """

    ids: set[int] = set()

    # findall 把文本里所有 [字段=数字] 形式的引用都揪出来，比如
    # "...[chunk_id=10, 12]..." 会匹配成 ("chunk_id", "10, 12")
    for matched_field, raw_ids in _CITATION_PATTERN.findall(text or ""):

        # 只留目标字段，别的字段（比如查 chunk_id 时遇到 people_id）直接跳过
        if matched_field != field:
            continue

        # 数字部分可能是逗号并列写法，比如 "10, 12"，先按逗号拆开
        for piece in raw_ids.split(","):

            # 去掉拆开后可能带的空格，" 12" → "12"
            piece = piece.strip()
            if piece.isdigit():
                ids.add(int(piece))

    return ids


def Validate_Citations(text: str, field: str, valid_ids: set[int]) -> str | None:
    """校验答案里某个字段的引用 id 是否全部来自合法证据池。

    Args:
        text: LLM 生成的答案文本。
        field: 引用字段名，"people_id" / "event_id" / "chunk_id"。
        valid_ids: 本轮真实证据池里出现过的合法 id 集合。
    Returns:
        校验通过返回 None；不通过返回一句错误描述，供调用方打日志、决定是否拒答。
    """

    cited = Extract_Cited_Ids(text, field)

    # 差集：答案引用了、但不在合法 id 里的部分
    bogus = cited - valid_ids

    # 差集为空（没有编造的 id）才算校验通过
    if not bogus:
        return None

    return f"引用了不存在于证据池的 {field}：{sorted(bogus)}（证据池合法 id：{sorted(valid_ids)}）"
