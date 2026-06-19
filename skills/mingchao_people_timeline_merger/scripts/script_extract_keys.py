"""
script_extract_keys.py
======================
用法：
    python script_extract_keys.py <目录路径>

功能：
    1. 扫描目录中所有 chunk_*_*_people.json 和 chunk_*_*_timeline.json
    2. 统计每个 key（人物名 / 事件名）出现在几个文件中
    3. PASSTHROUGH（仅出现 1 次）→ 直接写入
         passthrough_people.json
         passthrough_timeline.json
    4. MERGE（出现 2+ 次）→ 列入 merge_keys.txt
    5. 运行预冲突扫描（primary_identity / year / era 冲突）
    6. 输出摘要报告

输出文件（均写入 <目录路径>/）：
    merge_keys.txt         — 需要 LLM 合并的 key 列表
    passthrough_people.json    — 无需合并的人物条目（直接用作初始输出）
    passthrough_timeline.json  — 无需合并的事件条目
"""

import json
import os
import re
import sys
from collections import defaultdict


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def parse_chunk_range(filename):
    """从文件名中解析 (chunk_start, chunk_end)，失败返回 None。"""
    m = re.match(r"chunk_(\d+)_(\d+)_(people|timeline)\.json$", os.path.basename(filename))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scan_files(directory):
    """扫描目录，返回 (people_files, timeline_files)，按 chunk_start 升序排序。"""
    people_files = []
    timeline_files = []
    final_people = {"mingchao_people.json"}
    final_timeline = {"mingchao_timeline.json"}

    for fname in os.listdir(directory):
        parsed = parse_chunk_range(fname)
        if parsed is None:
            continue
        start, end, kind = parsed
        full_path = os.path.join(directory, fname)
        if kind == "people":
            people_files.append((start, end, full_path))
        else:
            timeline_files.append((start, end, full_path))

    people_files.sort(key=lambda x: (x[0], x[1]))
    timeline_files.sort(key=lambda x: (x[0], x[1]))
    return people_files, timeline_files


# ──────────────────────────────────────────────
# People 分析
# ──────────────────────────────────────────────

def analyze_people(people_files):
    """
    返回：
        key_sources: {name: [(start, end, path, record), ...]}
        conflicts:   [(name, field, val_a, file_a, val_b, file_b), ...]
    """
    key_sources = defaultdict(list)

    for start, end, path in people_files:
        data = load_json(path)
        if not isinstance(data, dict):
            print(f"  ⛔ {os.path.basename(path)} 顶层不是 dict，跳过", file=sys.stderr)
            continue
        for name, record in data.items():
            key_sources[name].append((start, end, path, record))

    # 预冲突扫描：primary_identity
    conflicts = []
    for name, sources in key_sources.items():
        if len(sources) < 2:
            continue
        identities = [(s[2], s[3].get("primary_identity", "")) for s in sources]
        ref_id = identities[0][1]
        for fpath, pid in identities[1:]:
            if pid and ref_id and pid != ref_id:
                conflicts.append((name, "primary_identity", ref_id, identities[0][0], pid, fpath))

    return key_sources, conflicts


# ──────────────────────────────────────────────
# Timeline 分析
# ──────────────────────────────────────────────

def analyze_timeline(timeline_files):
    """
    返回：
        key_sources: {event_name: [(start, end, path, record), ...]}
        conflicts:   [(event, field, val_a, file_a, val_b, file_b), ...]
    """
    key_sources = defaultdict(list)

    for start, end, path in timeline_files:
        data = load_json(path)
        if not isinstance(data, list):
            print(f"  ⛔ {os.path.basename(path)} 顶层不是 list，跳过", file=sys.stderr)
            continue
        for record in data:
            event = record.get("event", "")
            if not event:
                continue
            key_sources[event].append((start, end, path, record))

    # 预冲突扫描：year 和 era
    conflicts = []
    for event, sources in key_sources.items():
        if len(sources) < 2:
            continue
        ref = sources[0][3]
        ref_year = ref.get("year")
        ref_era  = ref.get("era", "")
        ref_path = sources[0][2]
        for _, _, fpath, rec in sources[1:]:
            y = rec.get("year")
            e = rec.get("era", "")
            if y and ref_year and y != ref_year:
                conflicts.append((event, "year", ref_year, ref_path, y, fpath))
            if e and ref_era and e != ref_era:
                # 忽略仅标点/空白差异
                if e.strip().rstrip("。") != ref_era.strip().rstrip("。"):
                    conflicts.append((event, "era", ref_era, ref_path, e, fpath))

    return key_sources, conflicts


# ──────────────────────────────────────────────
# 写出
# ──────────────────────────────────────────────

def write_passthrough_people(key_sources, directory):
    """将仅出现 1 次的人物直接写入 passthrough_people.json。"""
    out = {}
    for name, sources in sorted(key_sources.items(), key=lambda x: (x[1][0][0], x[0])):
        if len(sources) == 1:
            out[name] = sources[0][3]
    path = os.path.join(directory, "passthrough_people.json")
    dump_json(out, path)
    return len(out), path


def write_passthrough_timeline(key_sources, directory):
    """将仅出现 1 次的事件直接写入 passthrough_timeline.json（保持 list 格式）。"""
    # 按 year 升序，year 缺失排到最后
    items = []
    for event, sources in key_sources.items():
        if len(sources) == 1:
            items.append(sources[0][3])
    items.sort(key=lambda r: (r.get("year") or 9999, r.get("event", "")))
    path = os.path.join(directory, "passthrough_timeline.json")
    dump_json(items, path)
    return len(items), path


def write_merge_keys(people_key_sources, timeline_key_sources, directory):
    """写出 merge_keys.txt，包含 PEOPLE_MERGE 和 TIMELINE_MERGE 两节。"""
    people_merge = sorted(
        [name for name, s in people_key_sources.items() if len(s) >= 2]
    )
    timeline_merge = sorted(
        [event for event, s in timeline_key_sources.items() if len(s) >= 2]
    )

    path = os.path.join(directory, "merge_keys.txt")
    lines = []
    lines.append("# merge_keys.txt — 由 script_extract_keys.py 自动生成")
    lines.append("# 以下 key 在 2+ 个 chunk 文件中出现，需要 LLM 合并。")
    lines.append("")
    lines.append("[PEOPLE_MERGE]")
    for name in people_merge:
        lines.append(name)
    lines.append("")
    lines.append("[TIMELINE_MERGE]")
    for event in timeline_merge:
        lines.append(event)
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return people_merge, timeline_merge, path


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("用法: python script_extract_keys.py <目录路径>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"⛔ 目录不存在: {directory}")
        sys.exit(1)

    print(f"\n📂 扫描目录: {directory}\n")

    # ── 扫描文件 ──
    people_files, timeline_files = scan_files(directory)
    print(f"  发现 people  文件: {len(people_files)} 个")
    print(f"  发现 timeline 文件: {len(timeline_files)} 个")
    if not people_files and not timeline_files:
        print("⛔ 未找到任何 chunk_*_*_people.json 或 chunk_*_*_timeline.json")
        sys.exit(1)

    # ── 分析 ──
    people_sources, people_conflicts = analyze_people(people_files)
    timeline_sources, timeline_conflicts = analyze_timeline(timeline_files)

    # ── 预冲突报告 ──
    all_conflicts = people_conflicts + timeline_conflicts
    if all_conflicts:
        print("\n⛔ 预冲突扫描发现以下冲突，请裁决后再继续：\n")
        for item in all_conflicts:
            name, field, val_a, file_a, val_b, file_b = item
            print(f"  [{name}] 字段 {field} 冲突：")
            print(f"    {os.path.basename(file_a)} → {val_a}")
            print(f"    {os.path.basename(file_b)} → {val_b}")
        print()
        sys.exit(2)

    # ── 写出 pass-through ──
    n_pt_p, pt_p_path = write_passthrough_people(people_sources, directory)
    n_pt_t, pt_t_path = write_passthrough_timeline(timeline_sources, directory)

    # ── 写出 merge_keys.txt ──
    people_merge, timeline_merge, mk_path = write_merge_keys(
        people_sources, timeline_sources, directory
    )

    # ── 汇总报告 ──
    n_total_p  = len(people_sources)
    n_merge_p  = len(people_merge)
    n_total_t  = len(timeline_sources)
    n_merge_t  = len(timeline_merge)

    print("\n── 分析结果 ──────────────────────────────────")
    print(f"  People  总计 {n_total_p} 人：")
    print(f"    PASSTHROUGH（直接写出）: {n_pt_p} 人  → {os.path.basename(pt_p_path)}")
    print(f"    MERGE（需要 LLM）      : {n_merge_p} 人")
    if people_merge:
        for name in people_merge:
            n = len(people_sources[name])
            print(f"      · {name}（出现 {n} 次）")

    print(f"\n  Timeline 总计 {n_total_t} 件：")
    print(f"    PASSTHROUGH（直接写出）: {n_pt_t} 件  → {os.path.basename(pt_t_path)}")
    print(f"    MERGE（需要 LLM）      : {n_merge_t} 件")
    if timeline_merge:
        for event in timeline_merge:
            n = len(timeline_sources[event])
            print(f"      · {event}（出现 {n} 次）")

    print(f"\n  merge_keys.txt → {mk_path}")
    print("\n✅ 预冲突扫描通过，passthrough 文件已写出。")
    print("   下一步：将 merge_keys.txt 中的 key 传给 script_filter_by_key.py 进行批量合并。\n")


if __name__ == "__main__":
    main()
