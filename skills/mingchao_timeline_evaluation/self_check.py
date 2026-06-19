#!/usr/bin/env python3
"""Timeline 题库自检：keypoint 溯源验证 + 问题模板多样性检查。

用法：
    python3 self_check.py --eval_json <题库JSON> --timeline_json <时间线JSON>
"""

import argparse
import json
import re
from collections import Counter


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
    ("几次",   r"几次"),
    ("几年",   r"几年"),
    ("多少",   r"多少"),
]


def detect_q_word(question: str) -> str:
    for label, pattern in _Q_PATTERNS:
        if re.search(pattern, question):
            return label
    return "其他"


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_source(source_str: str, timeline_by_id: dict):
    """把 source path 解析成实际字段值，返回 (value, error_msg)。"""
    # timeline[EVENT_ID].FIELD
    m = re.match(r"timeline\[(\d+)\]\.(\w+)$", source_str)
    if m:
        eid, field = int(m.group(1)), m.group(2)
        ev = timeline_by_id.get(eid)
        if ev is None:
            return None, f"event_id={eid} 不在数据中"
        val = ev.get(field)
        if val is None:
            return None, f"event_id={eid} 没有字段 '{field}'"
        return val, None

    return None, f"无法解析 source 格式：{source_str}"


def text_in_value(text: str, value) -> bool:
    if isinstance(value, str):
        return text in value
    if isinstance(value, list):
        return any(text in str(item) for item in value)
    return text in str(value)


def part1_keypoint_tracing(entries: list, timeline_by_id: dict):
    print("=" * 68)
    print("Part 1  Keypoint 溯源验证  —  逐条确认 answer 能从 source 字段读出")
    print("=" * 68)

    total_kps = 0
    fail_kps  = 0

    for e in entries:
        print(f"\n[{e['qna_id']}]  {e['sub_type']}")
        print(f"问：{e['question']}")

        for kp in e.get("keypoints", []):
            total_kps += 1
            text   = kp.get("answer", "")
            source = kp.get("source", "")

            # 派生值：source 是列表（持续时长、先后比较）
            if isinstance(source, list):
                print(f"  keypoint: 「{text}」（派生值）")
                for s in source:
                    val, err = resolve_source(s, timeline_by_id)
                    if err:
                        print(f"    ⚠️  {s} → {err}")
                        fail_kps += 1
                    else:
                        print(f"    ✅  {s} → {str(val)[:60]}")
                continue

            val, err = resolve_source(source, timeline_by_id)
            if err:
                print(f"  ⚠️  [{source}] → {err}")
                fail_kps += 1
                continue

            preview = str(val)[:100].replace("\n", " ")
            found   = text_in_value(text, val)
            mark    = "✅" if found else "⚠️ "
            print(f"  {mark} 「{text}」")
            print(f"      source: {source}")
            print(f"      字段值: {preview}")
            if not found:
                print(f"      ^ answer 未在字段值中找到，请确认是否依赖先验知识或填写有误")
                fail_kps += 1

        print("  " + "─" * 62)

    print(f"\n共检查 {total_kps} 条 keypoint，问题条数 {fail_kps} 条")
    if fail_kps == 0:
        print("✅ Keypoint 溯源全部通过")
    else:
        print(f"⚠️  {fail_kps} 条需要修正，改完后重新运行")


def part2_diversity(entries: list):
    print("\n" + "=" * 68)
    print("Part 2  问题模板多样性检查")
    print("=" * 68)

    sub_types = {}
    for e in entries:
        st = e.get("sub_type", "unknown")
        sub_types.setdefault(st, []).append(e)

    all_words  = []
    violations = 0

    for st, group in sub_types.items():
        words = [detect_q_word(e["question"]) for e in group]
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
        print("✅ 多样性检查通过")
    else:
        print(f"⚠️  共 {violations} 处多样性违规，修改后重新运行")


def main():
    parser = argparse.ArgumentParser()
    _DEFAULT_TIMELINE = "/home/levizenith/SednaAI/RAG_Ming_Refine/data/people_timeline/mingchao_timeline.json"
    parser.add_argument("--eval_json",     required=True,   help="生成的题库 JSON 路径")
    parser.add_argument("--timeline_json", default=_DEFAULT_TIMELINE, help="时间线 JSON 路径（可选，默认项目标准路径）")
    args = parser.parse_args()

    entries         = load_json(args.eval_json)
    timeline_list   = load_json(args.timeline_json)
    timeline_by_id  = {e["event_id"]: e for e in timeline_list}

    part1_keypoint_tracing(entries, timeline_by_id)
    part2_diversity(entries)


if __name__ == "__main__":
    main()
