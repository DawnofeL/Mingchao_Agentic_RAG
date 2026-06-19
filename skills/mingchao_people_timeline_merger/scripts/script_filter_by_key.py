"""
script_filter_by_key.py
=======================
用法：
    # 提取人物（最多3个 key）
    python script_filter_by_key.py <目录路径> people <key1> [key2] [key3]

    # 提取事件（最多5个 key）
    python script_filter_by_key.py <目录路径> timeline <event1> [event2] ... [event5]

功能：
    从目录中所有 chunk_*_*_people.json / chunk_*_*_timeline.json 里，
    提取指定 key 在每个文件中对应的记录，写出 batch_input.json 供 LLM 合并。

输出文件（写入 <目录路径>/）：
    batch_input.json — 结构如下：

    People 模式：
    {
      "type": "people",
      "keys": ["朱元璋", "汤和"],
      "data": {
        "朱元璋": {
          "chunk_1_20_people.json":  { ...record... },
          "chunk_21_40_people.json": { ...record... }
        },
        "汤和": {
          "chunk_1_20_people.json":  { ...record... }
        }
      }
    }

    Timeline 模式：
    {
      "type": "timeline",
      "keys": ["靖难之役"],
      "data": {
        "靖难之役": {
          "chunk_171_220_timeline.json": { ...record... },
          "chunk_221_263_timeline.json": { ...record... }
        }
      }
    }

注意：
    - 只包含真正包含该 key 的文件，不出现空条目
    - LLM 读取 batch_input.json，完成合并后写出 batch_output.json（同目录）
    - batch_output.json 的格式与 batch_input.json 的 data 值相同：
        People   → { "朱元璋": { merged_record }, ... }
        Timeline → { "靖难之役": { merged_record }, ... }
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


def scan_files_by_kind(directory, kind):
    """返回 [(start, end, path), ...] 按 chunk_start 升序。"""
    results = []
    for fname in os.listdir(directory):
        parsed = parse_chunk_range(fname)
        if parsed is None:
            continue
        start, end, k = parsed
        if k == kind:
            results.append((start, end, os.path.join(directory, fname)))
    results.sort(key=lambda x: (x[0], x[1]))
    return results


# ──────────────────────────────────────────────
# People 提取
# ──────────────────────────────────────────────

def extract_people(directory, keys):
    files = scan_files_by_kind(directory, "people")
    data = {k: {} for k in keys}

    for start, end, path in files:
        raw = load_json(path)
        if not isinstance(raw, dict):
            continue
        fname = os.path.basename(path)
        for key in keys:
            if key in raw:
                data[key][fname] = raw[key]

    # 报告哪些 key 未找到任何记录
    missing = [k for k, v in data.items() if not v]
    if missing:
        print(f"  ⚠ 以下 key 在所有文件中均未找到：{missing}", file=sys.stderr)

    return data


# ──────────────────────────────────────────────
# Timeline 提取
# ──────────────────────────────────────────────

def extract_timeline(directory, keys):
    files = scan_files_by_kind(directory, "timeline")
    data = {k: {} for k in keys}

    for start, end, path in files:
        raw = load_json(path)
        if not isinstance(raw, list):
            continue
        fname = os.path.basename(path)
        for record in raw:
            event = record.get("event", "")
            if event in data:
                data[event][fname] = record

    missing = [k for k, v in data.items() if not v]
    if missing:
        print(f"  ⚠ 以下 event 在所有文件中均未找到：{missing}", file=sys.stderr)

    return data


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 4:
        print("用法:")
        print("  python script_filter_by_key.py <目录> people  <key1> [key2] [key3]")
        print("  python script_filter_by_key.py <目录> timeline <event1> [event2] ...")
        sys.exit(1)

    directory = sys.argv[1]
    mode      = sys.argv[2].lower()
    keys      = sys.argv[3:]

    if not os.path.isdir(directory):
        print(f"⛔ 目录不存在: {directory}")
        sys.exit(1)

    if mode not in ("people", "timeline"):
        print(f"⛔ 模式必须为 'people' 或 'timeline'，收到: {mode}")
        sys.exit(1)

    # key 数量检查
    max_keys = 3 if mode == "people" else 5
    if len(keys) > max_keys:
        print(f"⚠ {mode} 模式最多 {max_keys} 个 key，当前 {len(keys)} 个，截断处理。")
        keys = keys[:max_keys]

    print(f"\n📂 目录: {directory}")
    print(f"   模式: {mode}")
    print(f"   Key : {keys}\n")

    # 提取
    if mode == "people":
        data = extract_people(directory, keys)
    else:
        data = extract_timeline(directory, keys)

    # 写出
    output = {
        "type": mode,
        "keys": keys,
        "data": data
    }
    out_path = os.path.join(directory, "batch_input.json")
    dump_json(output, out_path)

    # 摘要
    print("── 提取结果 ──────────────────────────────────")
    for key in keys:
        sources = data.get(key, {})
        if sources:
            print(f"  {key}: 出现在 {len(sources)} 个文件 → {list(sources.keys())}")
        else:
            print(f"  {key}: ⚠ 未找到")
    print(f"\n✅ batch_input.json 已写出 → {out_path}")
    print("   LLM 完成合并后，请将结果写入同目录的 batch_output.json。\n")


if __name__ == "__main__":
    main()
