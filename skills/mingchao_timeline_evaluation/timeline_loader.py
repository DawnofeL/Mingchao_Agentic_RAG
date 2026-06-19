"""按 event_id 区间加载时间线数据，渲染成干净的纯文本输出给 agent 阅读。

使用方法：
    python timeline_loader.py --start 1 --end 30
    python timeline_loader.py --start 1 --end 30 --json_path /path/to/mingchao_timeline.json

设计目的：
    agent 在出题前需要看事件区间内的详细数据。
    直接读整个 JSON 容易越界（258 条全量），
    所以提供这个脚本按 event_id 切片返回，agent 只能从这里拿内容。
"""

import argparse
import json
from pathlib import Path


DEFAULT_TIMELINE_PATH = "/home/levizenith/SednaAI/RAG_Ming_Refine/data/people_timeline/mingchao_timeline.json"


def Load_Timeline_Range(start: int, end: int, json_path: str = DEFAULT_TIMELINE_PATH) -> list:
    if start > end:
        raise ValueError(f"start ({start}) 不能大于 end ({end})")
    with open(json_path, encoding="utf-8") as f:
        all_events = json.load(f)
    selected = [e for e in all_events if start <= e["event_id"] <= end]
    selected.sort(key=lambda e: e["event_id"])
    for e in selected:
        e.pop("source_chunks", None)
    return selected


def Render_Timeline(events: list) -> str:
    lines = []
    for e in events:
        lines.append("=" * 64)
        lines.append(f"event_id: {e['event_id']}  |  year: {e.get('year', '—')}  |  era: {e.get('era', '—')}")
        lines.append(f"event: {e.get('event', '—')}")
        lines.append("-" * 64)

        if e.get("tags"):
            lines.append(f"tags:          {' / '.join(e['tags'])}")
        if e.get("location"):
            lines.append(f"location:      {e['location']}")
        if e.get("participants"):
            lines.append(f"participants:  {' / '.join(e['participants'])}")
        if e.get("outcome"):
            lines.append(f"outcome: {e['outcome']}")
        if e.get("summary"):
            lines.append(f"summary: {e['summary'][:200]}")

        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="加载事件区间，渲染输出。")
    parser.add_argument("--start",     type=int, required=True, help="起始 event_id（含）")
    parser.add_argument("--end",       type=int, required=True, help="结束 event_id（含）")
    parser.add_argument("--json_path", default=DEFAULT_TIMELINE_PATH, help="timeline JSON 路径")
    args = parser.parse_args()

    events = Load_Timeline_Range(args.start, args.end, args.json_path)

    if not events:
        print(f"区间 [{args.start}, {args.end}] 内未找到任何事件。")
        return

    print(f"共加载 {len(events)} 条事件，event_id 范围 [{events[0]['event_id']}, {events[-1]['event_id']}]\n")
    print(Render_Timeline(events))


if __name__ == "__main__":
    main()
