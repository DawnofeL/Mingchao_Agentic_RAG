"""
Usage:
    python self_check.py <*_Eval_*.json>

Read-only validation for Mingchao LLM assessment outputs. This script must not
create, update, or delete any files.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def main(eval_path: Path) -> int:
    if "_Eval_" not in eval_path.name or eval_path.suffix.lower() != ".json":
        print("错误：自检对象必须是 *_Eval_*.json 最终文件。")
        return 1

    data = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("错误：Eval 文件顶层必须是数组。")
        return 1

    errors = []

    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            errors.append(f"[index={index}] 条目必须是对象")
            continue

        qid = entry.get("qna_id", f"index={index}")
        score = entry.get("score")
        if not isinstance(score, dict):
            errors.append(f"[{qid}] 缺少 score 字典")
            continue

        for key in ("0", "1"):
            if key not in score:
                errors.append(f"[{qid}] score 缺少键 '{key}'")
            elif not isinstance(score[key], int) or score[key] < 0:
                errors.append(f"[{qid}] score['{key}'] 必须是非负整数")

        keypoints = entry.get("keypoints", [])
        if not isinstance(keypoints, list) or not keypoints:
            errors.append(f"[{qid}] keypoints 必须是非空数组")
            continue

        # 每个 keypoint 必须有 score 字段（0 或 1）
        kp_hit  = 0
        kp_miss = 0
        for kp_idx, kp in enumerate(keypoints):
            if not isinstance(kp, dict):
                errors.append(f"[{qid}] keypoints[{kp_idx}] 必须是对象")
                continue
            kp_score = kp.get("score")
            if kp_score not in (0, 1):
                errors.append(
                    f"[{qid}] keypoints[{kp_idx}].score 必须是 0 或 1，"
                    f"实际为 {kp_score!r}"
                )
            elif kp_score == 1:
                kp_hit += 1
            else:
                kp_miss += 1

        # score["0"] / score["1"] 必须与 keypoint 逐条计数一致
        if score.get("0", 0) != kp_miss:
            errors.append(
                f"[{qid}] score['0']={score.get('0')} 与 keypoints 中 score=0 的数量 {kp_miss} 不一致"
            )
        if score.get("1", 0) != kp_hit:
            errors.append(
                f"[{qid}] score['1']={score.get('1')} 与 keypoints 中 score=1 的数量 {kp_hit} 不一致"
            )

    if errors:
        print("=== Part 1 / Part 2 失败 ===")
        for error in errors:
            print(" ", error)
        print(f"\n共 {len(errors)} 处错误。")
        return 1

    print(f"Part 1 & Part 2 通过，共 {len(data)} 题，结构完整。")
    print("\n=== Part 3 分布统计 ===")

    groups = defaultdict(
        lambda: {"hit": 0, "miss": 0, "total_kp": 0, "count": 0}
    )

    for entry in data:
        score = entry["score"]
        sub_type = entry.get("sub_type", "unknown")
        group = groups[sub_type]
        group["hit"] += score["1"]
        group["miss"] += score["0"]
        group["total_kp"] += score["0"] + score["1"]
        group["count"] += 1

    for sub_type, group in sorted(groups.items()):
        recall = group["hit"] / group["total_kp"] if group["total_kp"] else 0
        print(
            f"  {sub_type:12s}  题数={group['count']:3d}  "
            f"召回率={recall:.1%}  "
            f"(命中={group['hit']} 漏答={group['miss']})"
        )

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法：python self_check.py <*_Eval_*.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1])))
