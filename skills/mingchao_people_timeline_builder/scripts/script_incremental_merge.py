"""
增量合并脚本：将 LLM 从单个 chunk 提取的 patch 合并进主 JSON 文件。

设计原则：
  - 合并逻辑全在 Python 内完成，不向 stdout 输出完整 JSON（防止撑大上下文）
  - 只打印简短合并报告（3 行），让 LLM 知道合并结果即可
  - 支持从无到有（输出文件不存在时自动创建）
  - 所有写操作使用 UTF-8 + ensure_ascii=False，写完后重新解析验证

使用方式：
    python script_incremental_merge.py \\
        --people_patch  patch_people.json \\
        --timeline_patch patch_timeline.json \\
        --output_people  chunk_{start}_{end}_people.json \\
        --output_timeline chunk_{start}_{end}_timeline.json

patch 格式：
  patch_people.json   — 与 people.json 相同的顶层 key → value 结构，只含本轮新增/更新的人物
  patch_timeline.json — 与 timeline.json 相同的列表结构，只含本轮新增/更新的事件
"""

import argparse
import json
import os
import sys


# ──────────────────────────────────────────────────────────────
# I/O 工具
# ──────────────────────────────────────────────────────────────

def Load_Json(path, default):
    """读取 JSON 文件，文件不存在时返回 default。"""
    if not os.path.exists(path):
        return default
    with open(path, encoding = "utf-8") as f:
        return json.load(f)


def Save_Json(path, data):
    """用 UTF-8 写出 JSON，写完后重新解析验证无乱码。"""
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(data, f, ensure_ascii = False, indent = 2)
    # 验证：重新解析，确保无乱码、无非法字符
    with open(path, encoding = "utf-8") as f:
        raw = f.read()
    if "?" in raw and any(ord(c) > 127 for c in raw if c == "?"):
        print(f"[错误] {path} 写出后检测到可疑问号乱码，请检查！")
        sys.exit(1)
    json.loads(raw)  # 再次解析确认合法


# ──────────────────────────────────────────────────────────────
# 合并工具
# ──────────────────────────────────────────────────────────────

def Stable_Union(a: list, b: list) -> list:
    """稳定并集：保留首次出现顺序，去除完全重复项。"""
    seen   = set()
    result = []
    for item in a + b:
        key = (
            json.dumps(item, ensure_ascii = False, sort_keys = True)
            if isinstance(item, dict)
            else str(item)
        )
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def Merge_People(existing: dict, patch: dict) -> tuple:
    """
    将 patch 合并进 existing people dict。
    返回 (merged_dict, new_count, updated_count)。

    字段合并规则：
      aliases / secondary_identity / era / roles / events / relationships — 稳定并集
      primary_identity   — 以 patch 值覆盖（更晚的 chunk 代表最终/最主要身份）
      source_chunks      — 合并去重后按数值升序排序
      summary            — 分段拼接保留（用 [PART] 分隔），供 Step 3 最终合成时使用；
                           不取更长、不静默丢弃，确保各批时段内容均不丢失
    """
    new_count     = 0
    updated_count = 0

    for name, new_data in patch.items():
        if name not in existing:
            existing[name] = new_data
            new_count += 1
        else:
            old           = existing[name]
            updated_count += 1

            # 列表字段：稳定并集
            for field in ["aliases", "secondary_identity", "era", "roles", "events"]:
                old[field] = Stable_Union(old.get(field, []), new_data.get(field, []))

            # relationships：稳定并集（每条 relationship 是 dict，整体去重）
            old["relationships"] = Stable_Union(
                old.get("relationships", []),
                new_data.get("relationships", [])
            )

            # primary_identity：patch 值非空时覆盖
            if new_data.get("primary_identity"):
                old["primary_identity"] = new_data["primary_identity"]

            # source_chunks：合并、去重、升序
            old["source_chunks"] = sorted(
                set(old.get("source_chunks", []) + new_data.get("source_chunks", []))
            )

            # summary：分段拼接保留，严禁静默丢弃任何批次内容
            # [PART] 是批次分隔符，Step 3 最终合成时 LLM 会将其改写为统一摘要
            old_summary = old.get("summary", "")
            new_summary = new_data.get("summary", "")
            if new_summary and new_summary.strip() != old_summary.strip():
                if old_summary:
                    old["summary"] = old_summary + "\n[PART]\n" + new_summary
                else:
                    old["summary"] = new_summary

    return existing, new_count, updated_count


def Merge_Timeline(existing: list, patch: list) -> tuple:
    """
    将 patch 合并进 existing timeline list。
    返回 (merged_list, new_count, updated_count)。

    字段合并规则：
      tags / participants — 稳定并集
      year               — 保留已有值；若已有为空则取 patch 值；双方都有且不同 → 打印警告，保留已有值
      era / location / outcome — 已有非空则保留；已有为空则取 patch 值
      source_chunks      — 合并去重升序
      summary            — 分段拼接保留（用 [PART] 分隔），同 people summary 规则
    主合并键：event 字段完全一致
    """
    event_map     = {ev["event"]: ev for ev in existing}
    new_count     = 0
    updated_count = 0

    for new_ev in patch:
        event_name = new_ev.get("event", "")
        if not event_name:
            continue

        if event_name not in event_map:
            event_map[event_name] = new_ev
            new_count += 1
        else:
            old_ev        = event_map[event_name]
            updated_count += 1

            # year
            old_year = old_ev.get("year")
            new_year = new_ev.get("year")
            if old_year is None and new_year is not None:
                old_ev["year"] = new_year
            elif old_year is not None and new_year is not None and old_year != new_year:
                print(
                    f"[警告] 事件 '{event_name}' 的 year 冲突："
                    f"已有={old_year}，patch={new_year}，保留已有值"
                )

            # 标量字段：已有非空则保留，为空则补充
            for field in ["era", "location", "outcome"]:
                if not old_ev.get(field) and new_ev.get(field):
                    old_ev[field] = new_ev[field]

            # 列表字段：稳定并集
            for field in ["tags", "participants"]:
                old_ev[field] = Stable_Union(
                    old_ev.get(field, []),
                    new_ev.get(field, [])
                )

            # source_chunks：合并去重升序
            old_ev["source_chunks"] = sorted(
                set(old_ev.get("source_chunks", []) + new_ev.get("source_chunks", []))
            )

            # summary：分段拼接保留，严禁静默丢弃任何批次内容
            old_summary = old_ev.get("summary", "")
            new_summary = new_ev.get("summary", "")
            if new_summary and new_summary.strip() != old_summary.strip():
                if old_summary:
                    old_ev["summary"] = old_summary + "\n[PART]\n" + new_summary
                else:
                    old_ev["summary"] = new_summary

    # 最终按 year 升序排序，year 相同时按 source_chunks 最小值
    merged = sorted(
        event_map.values(),
        key = lambda e: (e.get("year") or 9999, min(e.get("source_chunks", [9999])))
    )
    return merged, new_count, updated_count


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────

def Main():
    """解析参数，执行合并，打印简短报告。"""
    parser = argparse.ArgumentParser(description = "增量合并 KG patch 进主 JSON 文件")
    parser.add_argument("--people_patch",    required = True,  help = "本轮 people patch 路径")
    parser.add_argument("--timeline_patch",  required = True,  help = "本轮 timeline patch 路径")
    parser.add_argument("--output_people",   required = True,  help = "主 people JSON 输出路径")
    parser.add_argument("--output_timeline", required = True,  help = "主 timeline JSON 输出路径")
    args = parser.parse_args()

    # 加载现有数据（文件不存在则从空开始）
    existing_people   = Load_Json(args.output_people,   {})
    existing_timeline = Load_Json(args.output_timeline, [])

    # 加载本轮 patch
    patch_people   = Load_Json(args.people_patch,   {})
    patch_timeline = Load_Json(args.timeline_patch, [])

    # 合并
    merged_people,   p_new, p_updated = Merge_People(existing_people,   patch_people)
    merged_timeline, t_new, t_updated = Merge_Timeline(existing_timeline, patch_timeline)

    # 写出
    Save_Json(args.output_people,   merged_people)
    Save_Json(args.output_timeline, merged_timeline)

    # 简短报告（控制 stdout，防止撑大上下文）
    print(f"people:   新增 {p_new} 人 / 更新 {p_updated} 人  → 当前共 {len(merged_people)} 人")
    print(f"timeline: 新增 {t_new} 件 / 更新 {t_updated} 件  → 当前共 {len(merged_timeline)} 件")
    print(f"写出完成: {args.output_people} | {args.output_timeline}")


if __name__ == "__main__":
    Main()
