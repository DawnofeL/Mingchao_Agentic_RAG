"""
卷级合并候选筛选脚本。

遍历新 JSON 条目，在旧 JSON 中查找候选，分三路处理：
  - 老条目：原封不动复制进 staged 文件
  - 新条目无候选：直接 append 进 staged 文件
  - 新条目有候选：连同候选写入 merge_workload.md，供 LLM 阅读处理

使用方式：
    python script_find_candidates.py \
        --old_people   <旧大 JSON people 路径> \
        --old_timeline <旧大 JSON timeline 路径> \
        --new_people   <新卷 people 路径> \
        --new_timeline <新卷 timeline 路径> \
        --output_dir   <输出目录> \
        [--output_prefix chunk_1_661]
"""

import json
import argparse
import os
import re
from difflib import SequenceMatcher


NAME_SIM_THRESHOLD = 0.65       # people name_similarity 兜底阈值
EVENT_SIM_THRESHOLD = 0.5       # timeline event 名相似度阈值
PARTICIPANTS_OVERLAP_MIN = 2    # timeline participants 重叠最小人数


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sim_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def extract_chunk_range(path):
    """从文件名里提取 chunk 起止编号，如 chunk_1_484_people.json → (1, 484)。"""
    m = re.search(r"chunk_(\d+)_(\d+)_", os.path.basename(path))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def find_people_candidates(new_key, new_entry, old_people):
    """在 old_people 中查找与 new_key/new_entry 可能是同一对象的候选。"""
    candidates = []
    new_aliases = set(new_entry.get("aliases", []))

    for old_key, old_entry in old_people.items():
        old_aliases = set(old_entry.get("aliases", []))
        hit_type = None

        if new_key == old_key:
            hit_type = "exact_key"
        elif new_key in old_aliases:
            hit_type = "key_in_aliases"
        elif old_key in new_aliases:
            hit_type = "old_key_in_new_aliases"
        elif new_aliases & old_aliases:
            hit_type = "aliases_intersection"
        elif sim_ratio(new_key, old_key) >= NAME_SIM_THRESHOLD:
            hit_type = "name_similarity"

        if hit_type:
            candidates.append((old_key, old_entry, hit_type))

    return candidates


def find_timeline_candidates(new_event_name, new_entry, old_timeline):
    """在 old_timeline 中查找与 new_entry 可能是同一事件的候选。"""
    candidates = []
    new_year = new_entry.get("year")
    new_participants = set(new_entry.get("participants", []))

    for old_entry in old_timeline:
        if old_entry.get("year") != new_year:
            continue

        old_event = old_entry.get("event", "")
        old_participants = set(old_entry.get("participants", []))
        hit_type = None

        if sim_ratio(new_event_name, old_event) >= EVENT_SIM_THRESHOLD:
            hit_type = "same_year_name_close"
        elif len(new_participants & old_participants) >= PARTICIPANTS_OVERLAP_MIN:
            hit_type = "same_year_participants_overlap"

        if hit_type:
            candidates.append((old_entry, hit_type))

    return candidates


def fmt_json(data):
    """格式化为 markdown 代码块。"""
    return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"


def write_workload(
    people_tasks,
    timeline_tasks,
    old_people_path,
    new_people_path,
    output_dir,
    counts,
):
    """把需要 LLM 处理的条目写成 merge_workload.md。"""
    lines = []
    lines.append("# 卷级合并工作清单\n")
    lines.append(f"**旧 JSON**：{old_people_path}")
    lines.append(f"**新 JSON**：{new_people_path}\n")

    lines.append("## 概要\n")
    lines.append(f"- 老人物保持不变：{counts['old_people']} 个")
    lines.append(f"- 老事件保持不变：{counts['old_timeline']} 个")
    lines.append(f"- 新人物无候选直接加入：{counts['direct_add_people']} 个")
    lines.append(f"- 新事件无候选直接加入：{counts['direct_add_timeline']} 个")
    lines.append(f"- 需 LLM 处理的人物合并：{len(people_tasks)} 个")
    lines.append(f"- 需 LLM 处理的事件合并：{len(timeline_tasks)} 个\n")
    lines.append("---\n")

    if people_tasks:
        lines.append("## 需 LLM 处理的人物合并\n")
        for idx, (new_key, new_entry, candidates) in enumerate(people_tasks, 1):
            lines.append(f"### {idx}. 新人物 \"{new_key}\"\n")
            lines.append("**新条目完整内容：**")
            lines.append(fmt_json({new_key: new_entry}))
            lines.append(f"\n**候选老条目（共 {len(candidates)} 个）：**\n")
            for ci, (old_key, old_entry, hit_type) in enumerate(candidates, 1):
                lines.append(f"候选 {idx}.{ci}　老 key=\"{old_key}\"，命中类型={hit_type}")
                lines.append(fmt_json({old_key: old_entry}))
            lines.append("\n**任务**：阅读以上条目，按 SKILL.md 的 LLM 合并守则判断与处理。\n")
            lines.append("---\n")

    if timeline_tasks:
        lines.append("## 需 LLM 处理的事件合并\n")
        for idx, (new_event_name, new_entry, candidates) in enumerate(timeline_tasks, 1):
            lines.append(f"### {idx}. 新事件 \"{new_event_name}\"\n")
            lines.append("**新条目完整内容：**")
            lines.append(fmt_json(new_entry))
            lines.append(f"\n**候选老条目（共 {len(candidates)} 个）：**\n")
            for ci, (old_entry, hit_type) in enumerate(candidates, 1):
                old_event = old_entry.get("event", "?")
                lines.append(f"候选 {idx}.{ci}　老 event=\"{old_event}\"，命中类型={hit_type}")
                lines.append(fmt_json(old_entry))
            lines.append("\n**任务**：阅读以上条目，按 SKILL.md 的 LLM 合并守则判断与处理。\n")
            lines.append("---\n")

    workload_path = os.path.join(output_dir, "merge_workload.md")
    with open(workload_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return workload_path


def main():
    parser = argparse.ArgumentParser(description="卷级合并候选筛选")
    parser.add_argument("--old_people",    required=True, help="旧大 JSON people 路径")
    parser.add_argument("--old_timeline",  required=True, help="旧大 JSON timeline 路径")
    parser.add_argument("--new_people",    required=True, help="新卷 people 路径")
    parser.add_argument("--new_timeline",  required=True, help="新卷 timeline 路径")
    parser.add_argument("--output_dir",    required=True, help="输出目录")
    parser.add_argument("--output_prefix", default=None,
                        help="输出文件名前缀，如 chunk_1_661。默认从文件名自动推导。")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 推导输出前缀
    prefix = args.output_prefix
    if not prefix:
        old_start, _ = extract_chunk_range(args.old_people)
        _, new_end   = extract_chunk_range(args.new_people)
        if old_start is not None and new_end is not None:
            prefix = f"chunk_{old_start}_{new_end}"
        else:
            prefix = "merged"

    old_people   = load_json(args.old_people)
    old_timeline = load_json(args.old_timeline)
    new_people   = load_json(args.new_people)
    new_timeline = load_json(args.new_timeline)

    # staged 从老条目初始化
    staged_people   = dict(old_people)
    staged_timeline = list(old_timeline)

    people_tasks   = []
    timeline_tasks = []
    direct_add_people   = 0
    direct_add_timeline = 0

    # 处理新人物
    for new_key, new_entry in new_people.items():
        candidates = find_people_candidates(new_key, new_entry, old_people)
        if candidates:
            people_tasks.append((new_key, new_entry, candidates))
        else:
            staged_people[new_key] = new_entry
            direct_add_people += 1

    # 处理新事件
    for new_entry in new_timeline:
        new_event_name = new_entry.get("event", "")
        candidates = find_timeline_candidates(new_event_name, new_entry, old_timeline)
        if candidates:
            timeline_tasks.append((new_event_name, new_entry, candidates))
        else:
            staged_timeline.append(new_entry)
            direct_add_timeline += 1

    # 写 staged 文件
    people_staged_path   = os.path.join(args.output_dir, f"{prefix}_people.staged.json")
    timeline_staged_path = os.path.join(args.output_dir, f"{prefix}_timeline.staged.json")
    dump_json(staged_people,   people_staged_path)
    dump_json(staged_timeline, timeline_staged_path)

    # 写 merge_workload.md
    workload_path = None
    if people_tasks or timeline_tasks:
        workload_path = write_workload(
            people_tasks,
            timeline_tasks,
            args.old_people,
            args.new_people,
            args.output_dir,
            counts={
                "old_people":           len(old_people),
                "old_timeline":         len(old_timeline),
                "direct_add_people":    direct_add_people,
                "direct_add_timeline":  direct_add_timeline,
            },
        )

    # 汇报
    print(
        f"候选人物 {len(people_tasks)} 对 / 候选事件 {len(timeline_tasks)} 对，"
        f"直接新增 {direct_add_people} 人 / {direct_add_timeline} 件，"
        f"staged 文件已写出"
    )
    print(f"  人物 staged  : {people_staged_path}")
    print(f"  事件 staged  : {timeline_staged_path}")
    if workload_path:
        print(f"  LLM 工作清单 : {workload_path}")
    else:
        print("  无需 LLM 处理（所有新条目均直接加入）")


if __name__ == "__main__":
    main()
