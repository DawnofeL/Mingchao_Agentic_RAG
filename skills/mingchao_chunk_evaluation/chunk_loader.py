"""按 chunk_id 区间加载明朝那些事儿原文，渲染成干净的纯文本输出给 agent 阅读。

使用方法：
    python chunk_loader.py --start 100 --end 110
    python chunk_loader.py --start 100 --end 110 --json_path /path/to/chunks.json

设计目的：
    agent 在出题前需要看 chunk 区间内的原文。
    直接让 agent 读整个 JSON 文件容易越界（读爆 1300+ 条），
    所以提供这个脚本按 id 切片返回，agent 只能从这里拿内容。
"""

import argparse
import json
from pathlib import Path


DEFAULT_CHUNK_PATH = "/home/levizenith/SednaAI/RAG_Ming_Refine/data/raw/mingchao_chunks.json"


def Load_Chunk_Range(chunk_start: int, chunk_end: int, json_path: str = DEFAULT_CHUNK_PATH) -> list:
    """读取指定 chunk_id 区间的所有 chunk。

    Args:
        chunk_start: 区间起点（含）。
        chunk_end:   区间终点（含）。
        json_path:   chunk JSON 文件路径。
    Returns:
        过滤后的 chunk 列表，已按 chunk_id 升序。
    """

    if chunk_start > chunk_end:
        raise ValueError(f"chunk_start ({chunk_start}) 不能大于 chunk_end ({chunk_end})")

    with open(json_path, encoding = "utf-8") as file:
        all_chunks = json.load(file)

    selected = [c for c in all_chunks if chunk_start <= c["chunk_id"] <= chunk_end]
    selected.sort(key = lambda c: c["chunk_id"])
    return selected


def Render_Chunks(chunks: list) -> str:
    """把 chunk 列表渲染成方便人/LLM 阅读的纯文本。

    Args:
        chunks: Load_Chunk_Range 的返回值。
    Returns:
        多行字符串，每个 chunk 一段，附 chunk_id / volume / chapter / section / content。
    """

    lines = []
    for c in chunks:
        lines.append("=" * 60)
        lines.append(f"chunk_id: {c['chunk_id']}  |  volume: {c['volume']}  |  {c['chapter']}")
        lines.append(f"section: {c['section']}")
        lines.append("-" * 60)
        lines.append(c["content"])
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description = "加载明朝那些事儿 chunk 区间，渲染输出。")
    parser.add_argument("--start", type = int, required = True, help = "起始 chunk_id（含）")
    parser.add_argument("--end",   type = int, required = True, help = "结束 chunk_id（含）")
    parser.add_argument("--json_path", default = DEFAULT_CHUNK_PATH, help = "chunk JSON 路径")
    args = parser.parse_args()

    chunks = Load_Chunk_Range(args.start, args.end, args.json_path)

    if not chunks:
        print(f"区间 [{args.start}, {args.end}] 内未找到任何 chunk。")
        return

    first_id = chunks[0]["chunk_id"]
    last_id  = chunks[-1]["chunk_id"]
    volumes  = sorted({c["volume"] for c in chunks})

    print(f"共加载 {len(chunks)} 个 chunk，实际范围 [{first_id}, {last_id}]，涉及卷 {volumes}\n")
    print(Render_Chunks(chunks))


if __name__ == "__main__":
    main()