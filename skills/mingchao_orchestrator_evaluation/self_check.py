#!/usr/bin/env python3
"""Orchestrator 题库自检。

在 people/timeline 自检的基础上加四项 orchestrator 专属检查：
    1. 跨域强制：expected_tasks 必须同时含 intention=people 和 timeline。
    2. task_id 一致性：每个 keypoint.task_id 必须在 expected_tasks 中；每个 task 至少 1 个 keypoint。
    3. source-intention 匹配：timeline task 的 keypoint source 必须以 timeline[ 开头，people 同理。
    4. 子类型拓扑：四种 sub_type 各自的 depends_on 拓扑必须合规。

沿用 people/timeline 的两项基础检查：
    5. keypoint 溯源（解析 source 路径到具体字段值）。
    6. 问题模板多样性（疑问词分布）。

用法：
    python3 self_check.py --eval_json <题库JSON>
    python3 self_check.py --eval_json <题库JSON> --people_json <人物JSON> --timeline_json <事件JSON>
"""

import argparse
import json
import re
from collections import Counter


DEFAULT_PEOPLE   = "/home/levizenith/SednaAI/RAG_Ming_Refine/data/people_timeline/mingchao_people.json"
DEFAULT_TIMELINE = "/home/levizenith/SednaAI/RAG_Ming_Refine/data/people_timeline/mingchao_timeline.json"


_Q_PATTERNS = [
    ("为什么", r"为什么"),
    ("为何",   r"为何"),
    ("怎样",   r"怎样"),
    ("怎么",   r"怎么"),
    ("如何",   r"如何"),
    ("哪里",   r"哪里"),
    ("哪些",   r"哪些"),
    ("哪",     r"哪."),
    ("什么",   r"什么"),
    ("谁",     r"谁"),
    ("几个",   r"几个"),
    ("多少",   r"多少"),
]


# 四种 sub_type 的拓扑约束。返回 (合规, 失败原因)
def Check_Topology(sub_type: str, tasks: list):
    n = len(tasks)
    deps = {t["task_id"]: t.get("depends_on", []) for t in tasks}

    if sub_type == "并行跨域":
        if n < 2:
            return False, f"并行跨域至少 2 个 task，当前 {n} 个"
        if any(deps[tid] for tid in deps):
            return False, "并行跨域所有 depends_on 必须为空"
        return True, ""

    if sub_type == "顺序两级":
        if n != 2:
            return False, f"顺序两级恰好 2 个 task，当前 {n} 个"
        if deps["t1"]:
            return False, "顺序两级 t1 不能有依赖"
        if deps.get("t2") != ["t1"]:
            return False, "顺序两级 t2 必须 depends_on=[t1]"
        return True, ""

    if sub_type == "扇出枚举":
        if n < 3:
            return False, f"扇出枚举至少 3 个 task（1 个 source + 2 个扇出），当前 {n} 个"
        if deps["t1"]:
            return False, "扇出枚举 t1 不能有依赖"
        fanout = [tid for tid in deps if tid != "t1"]
        if not all(deps[tid] == ["t1"] for tid in fanout):
            return False, "扇出枚举除 t1 外所有 task 必须 depends_on=[t1]"
        return True, ""

    if sub_type == "严格链式":
        if n < 3:
            return False, f"严格链式至少 3 个 task，当前 {n} 个"
        if deps["t1"]:
            return False, "严格链式 t1 不能有依赖"
        # t2 → t1, t3 → t2, t4 → t3...
        for i in range(2, n + 1):
            tid  = f"t{i}"
            prev = f"t{i-1}"
            if deps.get(tid) != [prev]:
                return False, f"严格链式 {tid} 必须 depends_on=[{prev}]"
        return True, ""

    return False, f"未知 sub_type：{sub_type}"


def Resolve_People_Source(source_str: str, people_by_name: dict):
    m = re.match(r"people\[(.+?)\]\.(\w+)$", source_str)
    if m:
        name, field = m.group(1), m.group(2)
        p = people_by_name.get(name)
        if p is None:
            return None, f"人物 '{name}' 不在数据中"
        val = p.get(field)
        if val is None:
            return None, f"人物 '{name}' 无字段 '{field}'"
        return val, None

    m = re.match(r"people\[(.+?)\]\.relationships\[(.+?)\]\.(\w+)$", source_str)
    if m:
        name, target, sub_field = m.group(1), m.group(2), m.group(3)
        p = people_by_name.get(name)
        if p is None:
            return None, f"人物 '{name}' 不在数据中"
        for rel in p.get("relationships", []):
            if rel.get("target") == target:
                val = rel.get(sub_field)
                if val is None:
                    return None, f"关系 [{name}→{target}] 无字段 '{sub_field}'"
                return val, None
        return None, f"人物 '{name}' 无与 '{target}' 的关系"

    return None, None  # 让上层判断是否为非 people 路径


def Resolve_Timeline_Source(source_str: str, timeline_by_id: dict):
    m = re.match(r"timeline\[(\d+)\]\.(\w+)$", source_str)
    if not m:
        return None, None
    event_id, field = int(m.group(1)), m.group(2)
    e = timeline_by_id.get(event_id)
    if e is None:
        return None, f"事件 id={event_id} 不存在"
    val = e.get(field)
    if val is None:
        return None, f"事件 id={event_id} 无字段 '{field}'"
    return val, None


def Resolve_Source(source_str: str, people_by_name: dict, timeline_by_id: dict):
    # 试 timeline 再试 people
    val, err = Resolve_Timeline_Source(source_str, timeline_by_id)
    if val is not None or err is not None:
        return val, err, "timeline"
    val, err = Resolve_People_Source(source_str, people_by_name)
    if val is not None or err is not None:
        return val, err, "people"
    return None, f"无法解析 source 格式：{source_str}", "unknown"


def Text_In_Value(text: str, value) -> bool:
    if isinstance(value, str):
        return text in value
    if isinstance(value, list):
        return any(text in str(item) for item in value)
    return text in str(value)


def Detect_Q_Word(question: str) -> str:
    for label, pattern in _Q_PATTERNS:
        if re.search(pattern, question):
            return label
    return "其他"


# ─────────── Part 1：结构性检查 ───────────


def Part1_Structure(entries: list):
    print("=" * 68)
    print("Part 1  跨域 + task 图 + source/intention 匹配  —  结构性硬约束")
    print("=" * 68)

    violations = 0

    for e in entries:
        qna_id   = e.get("qna_id", "?")
        sub_type = e.get("sub_type", "?")
        tasks    = e.get("expected_tasks", [])
        kps      = e.get("keypoints", [])
        print(f"\n[{qna_id}]  {sub_type}")

        if not tasks:
            print(f"  ⚠️  缺失 expected_tasks 字段")
            violations += 1
            continue

        # 检查 1：跨域强制
        intentions = {t.get("intention") for t in tasks}
        has_cross = "people" in intentions and "timeline" in intentions
        if has_cross:
            print(f"  ✅  跨域：intention 包含 people + timeline")
        else:
            print(f"  ⚠️  未跨域：intentions = {intentions}（必须同时含 people 和 timeline）")
            violations += 1

        # 检查 1.5：task_id 必须从 t1 连续编号
        expected_ids = [f"t{i}" for i in range(1, len(tasks) + 1)]
        actual_ids   = [t["task_id"] for t in tasks]
        if actual_ids != expected_ids:
            print(f"  ⚠️  task_id 编号不连续：{actual_ids}（应为 {expected_ids}）")
            violations += 1
            continue
        else:
            print(f"  ✅  task_id 从 t1 起连续编号")

        # 检查 2：task_id 一致性
        task_ids = {t["task_id"] for t in tasks}
        kp_task_ids = {kp.get("task_id") for kp in kps}
        orphan_kps = kp_task_ids - task_ids
        empty_tasks = task_ids - kp_task_ids
        if orphan_kps:
            print(f"  ⚠️  keypoint 引用了不存在的 task_id：{orphan_kps}")
            violations += 1
        if empty_tasks:
            print(f"  ⚠️  以下 task 没有任何 keypoint：{empty_tasks}")
            violations += 1
        if not orphan_kps and not empty_tasks:
            print(f"  ✅  task_id 一致：每个 task 至少 1 个 keypoint，所有 keypoint 归属合法")

        # 检查 3：depends_on 合法性
        seen_ids = []
        dep_ok = True
        for t in tasks:
            tid  = t["task_id"]
            for dep in t.get("depends_on", []):
                if dep not in seen_ids:
                    print(f"  ⚠️  {tid} 引用了未在之前出现的 task_id：{dep}")
                    violations += 1
                    dep_ok = False
            seen_ids.append(tid)
        if dep_ok:
            print(f"  ✅  depends_on 拓扑合法：无前向引用")

        # 检查 4：sub_type 拓扑合规
        ok, reason = Check_Topology(sub_type, tasks)
        if ok:
            print(f"  ✅  sub_type '{sub_type}' 拓扑合规")
        else:
            print(f"  ⚠️  sub_type '{sub_type}' 拓扑违规：{reason}")
            violations += 1

        # 检查 5：source-intention 匹配
        intention_map = {t["task_id"]: t.get("intention") for t in tasks}
        source_match_ok = True
        for kp in kps:
            tid = kp.get("task_id")
            intent = intention_map.get(tid)
            source = kp.get("source", "")
            srcs = source if isinstance(source, list) else [source]
            for s in srcs:
                if intent == "people" and not s.startswith("people["):
                    print(f"  ⚠️  task {tid} intention=people 但 keypoint source 不是 people[...]：{s}")
                    violations += 1
                    source_match_ok = False
                elif intent == "timeline" and not s.startswith("timeline["):
                    print(f"  ⚠️  task {tid} intention=timeline 但 keypoint source 不是 timeline[...]：{s}")
                    violations += 1
                    source_match_ok = False
        if source_match_ok and kps:
            print(f"  ✅  source 域与 task intention 全部匹配")

    print()
    if violations == 0:
        print("✅ Part 1 结构性检查全部通过")
    else:
        print(f"⚠️  Part 1 共 {violations} 处违规，必须修复")
    return violations


# ─────────── Part 2：keypoint 溯源 ───────────


def Part2_Keypoint_Tracing(entries: list, people_by_name: dict, timeline_by_id: dict):
    print("\n" + "=" * 68)
    print("Part 2  Keypoint 溯源验证  —  逐条确认 answer 能从 source 字段读出")
    print("=" * 68)

    total_kps = 0
    fail_kps  = 0

    for e in entries:
        print(f"\n[{e['qna_id']}]  {e['sub_type']}")
        print(f"问：{e['question']}")

        for kp in e.get("keypoints", []):
            total_kps += 1
            text   = kp.get("answer", "")
            tid    = kp.get("task_id", "?")
            source = kp.get("source", "")

            # source 是列表（派生值，如先后比较）
            if isinstance(source, list):
                print(f"  ({tid}) keypoint: 「{text}」")
                for s in source:
                    val, err, _ = Resolve_Source(s, people_by_name, timeline_by_id)
                    if err:
                        print(f"    ⚠️  {s} → {err}")
                        fail_kps += 1
                    else:
                        preview = str(val)[:80].replace("\n", " ")
                        print(f"    ✅  {s} → {preview}")
                continue

            val, err, _ = Resolve_Source(source, people_by_name, timeline_by_id)
            if err:
                print(f"  ⚠️  ({tid}) [{source}] → {err}")
                fail_kps += 1
                continue

            preview = str(val)[:100].replace("\n", " ")
            found   = Text_In_Value(text, val)
            mark    = "✅" if found else "⚠️ "
            print(f"  {mark} ({tid}) 「{text}」")
            print(f"      source: {source}")
            print(f"      字段值: {preview}")
            if not found:
                print(f"      ^ answer 未在字段值中找到，请确认是否依赖先验知识或填写有误")
                fail_kps += 1

        print("  " + "─" * 62)

    print(f"\n共检查 {total_kps} 条 keypoint，问题条数 {fail_kps} 条")
    if fail_kps == 0:
        print("✅ Part 2 Keypoint 溯源全部通过")
    else:
        print(f"⚠️  Part 2 {fail_kps} 条需要修正")
    return fail_kps


# ─────────── Part 3：问题模板多样性 ───────────


def Part3_Diversity(entries: list):
    print("\n" + "=" * 68)
    print("Part 3  问题模板多样性检查")
    print("=" * 68)

    sub_types = {}
    for e in entries:
        st = e.get("sub_type", "unknown")
        sub_types.setdefault(st, []).append(e)

    all_words  = []
    violations = 0

    for st, group in sub_types.items():
        words = [Detect_Q_Word(e["question"]) for e in group]
        all_words.extend(words)
        print(f"\n{st}（{len(group)} 题）：")
        for e, w in zip(group, words):
            print(f"  {e['qna_id']}  [{w}]  {e['question'][:50]}")

        unique = set(words)
        if len(unique) == 1 and len(words) > 1:
            violations += 1
            print(f"  ⚠️  该子类型 {len(words)} 题全部用「{words[0]}」句型，严重模板化，必须修改")
        else:
            print(f"  ✅ 句型分布：{dict((w, words.count(w)) for w in unique)}")

    if all_words:
        print("\n全局疑问词分布：")
        counter = Counter(all_words)
        total   = len(all_words)
        for word, cnt in counter.most_common():
            pct = cnt / total * 100
            flag = " ⚠️  占比过高" if pct >= 70 else ""
            print(f"  {word}：{cnt}/{total} ({pct:.0f}%){flag}")
            if pct >= 70:
                violations += 1

    print()
    if violations == 0:
        print("✅ Part 3 多样性检查通过")
    else:
        print(f"⚠️  Part 3 共 {violations} 处违规")
    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json",     required=True, help="生成的题库 JSON 路径")
    parser.add_argument("--people_json",   default=DEFAULT_PEOPLE,   help="人物 JSON 路径")
    parser.add_argument("--timeline_json", default=DEFAULT_TIMELINE, help="事件 JSON 路径")
    args = parser.parse_args()

    with open(args.eval_json,     encoding="utf-8") as f: entries  = json.load(f)
    with open(args.people_json,   encoding="utf-8") as f: people   = json.load(f)
    with open(args.timeline_json, encoding="utf-8") as f: timeline = json.load(f)

    people_by_name = {p["name"]: p for p in people}
    timeline_by_id = {e["event_id"]: e for e in timeline}

    v1 = Part1_Structure(entries)
    v2 = Part2_Keypoint_Tracing(entries, people_by_name, timeline_by_id)
    v3 = Part3_Diversity(entries)

    total = v1 + v2 + v3
    print("\n" + "=" * 68)
    if total == 0:
        print(f"✅ 三部分自检全部通过（共 {len(entries)} 道题）")
    else:
        print(f"⚠️  共 {total} 处违规需要修复，改完重新运行")
    print("=" * 68)


if __name__ == "__main__":
    main()
