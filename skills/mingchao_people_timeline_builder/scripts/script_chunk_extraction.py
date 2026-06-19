"""
从 chunk JSON 文件中提取指定 chunk_id 的原文文本，打印到 stdout。
设计原则：只输出原文文本，不输出其他元数据，最小化 token 消耗。

使用方式：
    python script_chunk_extraction.py <chunk_json_path> <chunk_id>

示例：
    python script_chunk_extraction.py ../chunks.json 1
"""

import json
import sys


def Main():
    """主函数：读取参数，定位 chunk，打印原文。"""
    if len(sys.argv) != 3:
        print("使用方式: python script_chunk_extraction.py <chunk_json_path> <chunk_id>")
        sys.exit(1)

    chunk_json_path = sys.argv[1]
    chunk_id        = int(sys.argv[2])

    with open(chunk_json_path, encoding = "utf-8") as f:
        all_chunks = json.load(f)

    target = None
    for c in all_chunks:
        cid = c.get("chunk_id", c.get("id"))
        if cid == chunk_id:
            target = c
            break

    if target is None:
        print(f"[错误] chunk_id={chunk_id} 不存在于 {chunk_json_path}")
        sys.exit(1)

    # 尝试常见的文本字段名
    text = (
        target.get("text")
        or target.get("content")
        or target.get("chunk_text")
        or target.get("passage")
        or ""
    )

    if not text:
        print(f"[警告] chunk_id={chunk_id} 文本字段为空，已知字段: {list(target.keys())}")
        sys.exit(1)

    print(f"=== chunk_id={chunk_id} ===")
    print(text)


if __name__ == "__main__":
    Main()
