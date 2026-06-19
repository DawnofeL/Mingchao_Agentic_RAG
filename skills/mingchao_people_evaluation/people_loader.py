"""按 people_id 区间加载人物数据，渲染成干净的纯文本输出给 agent 阅读。

使用方法：
    python people_loader.py --start 1 --end 20
    python people_loader.py --start 1 --end 20 --json_path /path/to/mingchao_people.json

设计目的：
    agent 在出题前需要看人物区间内的详细数据。
    直接读整个 JSON 容易越界（185 条全量），
    所以提供这个脚本按 people_id 切片返回，agent 只能从这里拿内容。
"""

import argparse
import json
from pathlib import Path


DEFAULT_PEOPLE_PATH = "/home/levizenith/SednaAI/RAG_Ming_Refine/data/people_timeline/mingchao_people.json"


def Load_People_Range(start: int, end: int, json_path: str = DEFAULT_PEOPLE_PATH) -> list:
    if start > end:
        raise ValueError(f"start ({start}) 不能大于 end ({end})")
    with open(json_path, encoding="utf-8") as f:
        all_people = json.load(f)
    selected = [p for p in all_people if start <= p["people_id"] <= end]
    selected.sort(key=lambda p: p["people_id"])
    for p in selected:
        p.pop("source_chunks", None)
    return selected


def Render_People(people: list) -> str:
    lines = []
    for p in people:
        lines.append("=" * 64)
        lines.append(f"people_id: {p['people_id']}  |  name: {p['name']}")
        lines.append("-" * 64)

        if p.get("aliases"):
            lines.append(f"aliases:           {' / '.join(p['aliases'])}")
        lines.append(f"primary_identity:  {p.get('primary_identity', '—')}")
        if p.get("secondary_identity"):
            lines.append(f"secondary_identity:{' / '.join(p['secondary_identity'])}")
        if p.get("era"):
            lines.append(f"era:               {' / '.join(p['era'])}")

        if p.get("roles"):
            lines.append("roles:")
            for r in p["roles"]:
                lines.append(f"  · {r}")

        if p.get("relationships"):
            lines.append("relationships:")
            for rel in p["relationships"]:
                ctx = rel.get("context", "")[:120]
                lines.append(f"  [{rel['type']}] {rel['target']} — {ctx}")

        if p.get("events"):
            lines.append("events:")
            for ev in p["events"][:8]:
                lines.append(f"  · {ev}")
            if len(p.get("events", [])) > 8:
                lines.append(f"  （共 {len(p['events'])} 条，仅展示前 8）")

        if p.get("summary"):
            lines.append(f"summary: {p['summary'][:200]}")

        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="加载人物区间，渲染输出。")
    parser.add_argument("--start",     type=int, required=True, help="起始 people_id（含）")
    parser.add_argument("--end",       type=int, required=True, help="结束 people_id（含）")
    parser.add_argument("--json_path", default=DEFAULT_PEOPLE_PATH, help="people JSON 路径")
    args = parser.parse_args()

    people = Load_People_Range(args.start, args.end, args.json_path)

    if not people:
        print(f"区间 [{args.start}, {args.end}] 内未找到任何人物。")
        return

    print(f"共加载 {len(people)} 位人物，people_id 范围 [{people[0]['people_id']}, {people[-1]['people_id']}]\n")
    print(Render_People(people))


if __name__ == "__main__":
    main()
