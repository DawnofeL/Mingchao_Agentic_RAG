"""
本脚本对 mingchao-kg-builder skill 生成的两个 JSON 文件进行结构校验。
只检查结构完整性、数据类型和交叉引用一致性，不检查词汇内容是否在某个列表中。

使用方式：
    python validate_kg.py chunk_1_91_people.json chunk_1_91_timeline.json
"""

import json
import sys


VALID_PRIMARY_IDENTITIES = {
    "皇帝", "皇室", "武将", "反叛势力", "文臣", "宦官",
    "社会人员", "清", "朝鲜", "日本", "外国",
}

VALID_TYPE_TAGS = {
    "战役", "起义", "政治", "外交", "人物节点", "民生", "其他",
}


def Validate_People(people_path):
    """校验 people.json 的结构完整性和字段格式。"""
    errors = []

    with open(people_path, encoding = "utf-8") as f:
        people = json.load(f)

    required_fields = {"aliases", "primary_identity", "secondary_identity", "era", "roles", "relationships", "events", "source_chunks", "summary"}
    rel_required = {"type", "target", "context"}

    for name, data in people.items():

        # 必填字段检查
        missing = required_fields - set(data.keys())
        if missing:
            errors.append(f"[{name}] 缺少字段: {missing}")
            continue

        # primary_identity 必须是固定枚举中的一个值
        pid = data.get("primary_identity", "")
        if not pid:
            errors.append(f"[{name}] primary_identity 为空，必须从固定枚举中选一个值")
        elif pid not in VALID_PRIMARY_IDENTITIES:
            errors.append(f"[{name}] primary_identity '{pid}' 不在合法枚举中: {sorted(VALID_PRIMARY_IDENTITIES)}")

        # secondary_identity 必须是列表（可以为空列表）
        if not isinstance(data.get("secondary_identity"), list):
            errors.append(f"[{name}] secondary_identity 必须是列表（无附加身份时填 []）")

        # era 必须是列表
        if not isinstance(data.get("era"), list):
            errors.append(f"[{name}] era 必须是列表")
        else:
            # era 不能包含明显的非年号描述词
            bad_era_words = ["早年", "晚年", "元末", "明初", "明末", "前期", "后期"]
            for e in data["era"]:
                if any(bad in e for bad in bad_era_words):
                    errors.append(f"[{name}] era 包含非年号描述词: '{e}'（应填官方年号如'洪武'）")

        # roles 不能为空
        if not data.get("roles"):
            errors.append(f"[{name}] roles 为空，至少填 2 项具体角色")

        # aliases 不应包含规范名本身
        if name in data.get("aliases", []):
            errors.append(f"[{name}] aliases 中包含了规范名本身")

        # relationships 格式
        for i, rel in enumerate(data.get("relationships", [])):
            missing_rel = rel_required - set(rel.keys())
            if missing_rel:
                errors.append(f"[{name}] relationships[{i}] 缺少字段: {missing_rel}")
                continue

            if not rel.get("type"):
                errors.append(f"[{name}] relationships[{i}].type 为空")

            if not rel.get("target"):
                errors.append(f"[{name}] relationships[{i}].target 为空")

            # context 必须有实质内容：禁止空洞表述，最短 12 字
            context = rel.get("context", "")
            if len(context) < 12:
                errors.append(f"[{name}] relationships[{i}].context 过短（当前 {len(context)} 字），需说明具体场景")

            bad_context = ["同为文臣", "同为将领", "同时期活动", "两人都"]
            for bad in bad_context:
                if bad in context:
                    errors.append(f"[{name}] relationships[{i}].context 含空洞表述: '{bad}'")

        # source_chunks 必须是整数列表
        for c in data.get("source_chunks", []):
            if not isinstance(c, int):
                errors.append(f"[{name}] source_chunks 包含非整数: {c}")

    return errors


def Validate_Timeline(timeline_path):
    """校验 timeline.json 的结构完整性和字段格式。"""
    errors = []

    with open(timeline_path, encoding = "utf-8") as f:
        timeline = json.load(f)

    required_fields = {"year", "era", "event", "tags", "location", "participants", "outcome", "source_chunks", "summary"}

    for i, ev in enumerate(timeline):
        label = f"事件[{i}]({ev.get('event', '?')})"

        # 必填字段
        missing = required_fields - set(ev.keys())
        if missing:
            errors.append(f"[{label}] 缺少字段: {missing}")
            continue

        # year 必须是整数
        if not isinstance(ev["year"], int):
            errors.append(f"[{label}] year 必须是整数，当前: {ev['year']}")

        # era 必须是完整年号纪年（不能只写年号单字，至少4字）
        era = ev.get("era", "")
        if not era:
            errors.append(f"[{label}] era 为空，应填完整年号纪年如'至正二十三年'")
        elif len(era) < 4:
            errors.append(f"[{label}] era 过短: '{era}'，应为完整年号纪年如'至正二十三年'")

        # tags 必须是非空列表
        tags = ev.get("tags", [])
        if not isinstance(tags, list) or len(tags) == 0:
            errors.append(f"[{label}] tags 必须是非空列表")
        else:
            # tags[0] 必须是固定类型枚举
            if tags[0] not in VALID_TYPE_TAGS:
                errors.append(
                    f"[{label}] tags[0] '{tags[0]}' 不是合法类型标签，"
                    f"必须从以下选一个: {sorted(VALID_TYPE_TAGS)}"
                )

            # tags 不应包含明显的年号/时代描述词（这些信息在 era/year 字段已有）
            bad_tag_patterns = ["早年", "晚年", "年间", "元末", "明初", "明末", "前期", "后期", "时期"]
            for tag in tags:
                for bad in bad_tag_patterns:
                    if bad in tag:
                        errors.append(f"[{label}] tag '{tag}' 含时代描述词 '{bad}'，改用 era/year 字段，tags 只填分类和系列名")
                        break

        # 序数 tag 格式检查（如果用了序数）
        for tag in tags:
            if tag.startswith("序数:"):
                try:
                    n = int(tag.split(":")[1])
                    if n < 1:
                        raise ValueError
                except (IndexError, ValueError):
                    errors.append(f"[{label}] 序数 tag 格式错误: '{tag}'，应为'序数:N'（N 为正整数）")

        # outcome 不能是空洞表述，最短 15 字
        outcome = ev.get("outcome", "")
        if len(outcome) < 15:
            errors.append(f"[{label}] outcome 过短（当前 {len(outcome)} 字），需包含：谁、做了什么、直接结果")

        bad_outcome = ["具有重要历史意义", "影响深远", "意义重大", "深远影响", "具有深远"]
        for bad in bad_outcome:
            if bad in outcome:
                errors.append(f"[{label}] outcome 含空洞表述: '{bad}'，改写为具体结果")

        # location 不能为空
        if not ev.get("location"):
            errors.append(f"[{label}] location 为空")

        # source_chunks 必须是整数列表
        for c in ev.get("source_chunks", []):
            if not isinstance(c, int):
                errors.append(f"[{label}] source_chunks 包含非整数: {c}")

    return errors


def Validate_Cross_Reference(people_path, timeline_path):
    """校验 people.events 与 timeline.event 的命名一致性，以及参与者/关系目标的存在性。"""
    errors = []

    with open(people_path, encoding = "utf-8") as f:
        people = json.load(f)
    with open(timeline_path, encoding = "utf-8") as f:
        timeline = json.load(f)

    timeline_event_names = {ev["event"] for ev in timeline}
    people_names = set(people.keys())

    # people.events 中的事件名必须在 timeline 中存在
    for name, data in people.items():
        for ev_name in data.get("events", []):
            if ev_name not in timeline_event_names:
                errors.append(f"交叉引用: [{name}].events 中 '{ev_name}' 与 timeline 中无完全一致的 event 名")

    # timeline.participants 中的人名必须在 people 中存在
    for ev in timeline:
        for p in ev.get("participants", []):
            if p not in people_names:
                errors.append(f"交叉引用: 事件 '{ev['event']}' 的 participant '{p}' 未在 people 中")

    # relationships.target 必须在 people 中存在
    for name, data in people.items():
        for rel in data.get("relationships", []):
            t = rel.get("target", "")
            if t and t not in people_names:
                errors.append(f"交叉引用: [{name}].relationships.target '{t}' 未在 people 中（使用规范名，不要用别名）")

    return errors


def Main():
    """主函数：读取路径参数，执行三项校验，输出结果。"""
    if len(sys.argv) != 3:
        print("使用方式: python validate_kg.py <people.json路径> <timeline.json路径>")
        sys.exit(1)

    people_path = sys.argv[1]
    timeline_path = sys.argv[2]

    print(f"校验: {people_path}")
    print(f"      {timeline_path}")
    print()

    all_errors = (
        Validate_People(people_path)
        + Validate_Timeline(timeline_path)
        + Validate_Cross_Reference(people_path, timeline_path)
    )

    if all_errors:
        print(f"❌ 发现 {len(all_errors)} 个问题：")
        print()
        for err in all_errors:
            print(f"  • {err}")
        print()
        print("修复以上问题后再继续。")
        sys.exit(1)

    with open(people_path, encoding = "utf-8") as f:
        people_count = len(json.load(f))
    with open(timeline_path, encoding = "utf-8") as f:
        timeline_count = len(json.load(f))

    print(f"✅ 全部通过")
    print(f"   people:   {people_count} 个人物")
    print(f"   timeline: {timeline_count} 个事件")
    sys.exit(0)


if __name__ == "__main__":
    Main()
