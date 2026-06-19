"""按 era 加载人物与事件的跨域视图，渲染成给 agent 阅读的纯文本。

使用方法：
    python orchestrator_loader.py --era 永乐
    python orchestrator_loader.py --era 建文,永乐
    python orchestrator_loader.py --era 洪武 --people_json /path --timeline_json /path

设计目的：
    orchestrator 题目必须跨 people 与 timeline 两域，
    用 era 作为统一锚点，自然保证两份数据有时段交叠。
    输出分两段：先列该 era 内全部人物，再列该 era 内全部事件，
    agent 据此设计跨域多任务问题。
"""

import argparse
import json


DEFAULT_PEOPLE_PATH   = "/home/levizenith/SednaAI/RAG_Ming_Refine/data/people_timeline/mingchao_people.json"
DEFAULT_TIMELINE_PATH = "/home/levizenith/SednaAI/RAG_Ming_Refine/data/people_timeline/mingchao_timeline.json"


def Parse_Eras(era_arg: str) -> list:
    # 用户可传 "永乐" 或 "建文,永乐"，统一返回去重 list
    eras = [e.strip() for e in era_arg.split(",") if e.strip()]
    if not eras:
        raise ValueError("--era 不能为空")
    return list(dict.fromkeys(eras))


def Load_People_By_Era(eras: list, json_path: str = DEFAULT_PEOPLE_PATH) -> list:
    with open(json_path, encoding="utf-8") as f:
        all_people = json.load(f)
    # people.era 是 list，任一年号命中即保留
    selected = [
        p for p in all_people
        if isinstance(p.get("era"), list)
        and any(era in p["era"] for era in eras)
    ]
    selected.sort(key=lambda p: p["people_id"])
    for p in selected:
        p.pop("source_chunks", None)
    return selected


def Load_Timeline_By_Era(eras: list, json_path: str = DEFAULT_TIMELINE_PATH) -> list:
    with open(json_path, encoding="utf-8") as f:
        all_events = json.load(f)
    # timeline.era 是字符串，含年号+序数（"永乐三年"），用子串匹配
    selected = [
        e for e in all_events
        if isinstance(e.get("era"), str)
        and any(era in e["era"] for era in eras)
    ]
    selected.sort(key=lambda e: e["event_id"])
    for e in selected:
        e.pop("source_chunks", None)
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
    parser = argparse.ArgumentParser(description="按 era 加载人物与事件跨域视图。")
    parser.add_argument("--era",           required=True, help="年号锚点，单值或逗号分隔多值（如 '永乐' 或 '建文,永乐'）")
    parser.add_argument("--people_json",   default=DEFAULT_PEOPLE_PATH,   help="人物 JSON 路径")
    parser.add_argument("--timeline_json", default=DEFAULT_TIMELINE_PATH, help="事件 JSON 路径")
    args = parser.parse_args()

    eras = Parse_Eras(args.era)
    people = Load_People_By_Era(eras, args.people_json)
    events = Load_Timeline_By_Era(eras, args.timeline_json)

    print(f"era 锚点：{' / '.join(eras)}")
    print(f"共加载 {len(people)} 位人物 + {len(events)} 条事件\n")

    if not people or not events:
        print("⚠️  人物或事件为空，无法生成跨域题。请换 era 或扩大范围。")
        return

    print("┌─────────────────────────────────────────────────────────────┐")
    print("│   PEOPLE  人物视图                                          │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(Render_People(people))

    print("┌─────────────────────────────────────────────────────────────┐")
    print("│   TIMELINE  事件视图                                        │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(Render_Timeline(events))


if __name__ == "__main__":
    main()
