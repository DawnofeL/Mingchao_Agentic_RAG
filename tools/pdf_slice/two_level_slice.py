"""两级切分总入口：把多卷 PDF 串联成一步到位的最终 chunk JSON。

`Two_Level_Slice` 把 `Slice_Single_Volume`（一级章节切分）和
`Langchain_Recursive_Slice`（二级 token 切分）串起来，
输入一组按卷号顺序排好的 PDF，直接落盘成最终合并 JSON。
"""

import json
from pathlib import Path

from tqdm import tqdm

from .ming_volume_slice import Slice_Single_Volume
from .langchain_recursive_slice import Langchain_Recursive_Slice


def Two_Level_Slice(
    pdf_paths: list[str],
    output_path: str,
    tokenizer,
    chunk_size: int = 1000,
    overlap_tokens: int = 50,
) -> dict:
    """串联一级章节切分和二级 token 切分，把多卷 PDF 直接处理成最终 chunk JSON。

    Args:
        pdf_paths: 按卷号顺序排列的 PDF 路径列表，第一个就是第一卷。
        output_path: 最终合并 JSON 的输出路径。
        tokenizer: 二级切分用的 tokenizer。
        chunk_size: 二级切分每块目标 token 数，默认 1000。
        overlap_tokens: overlap 的 token 预算，默认 50。
    Returns:
        dict，包含每卷 chunk 数、最终输出路径和总 chunk 数。
    """

    # 第一级：按列表顺序逐卷切，卷号直接用列表序号，不依赖文件名里的尾号
    all_chunks = []
    volume_counts = {}
    for volume_index, pdf_path in enumerate(tqdm(pdf_paths, desc = "一级切分"), start = 1):
        chunks = Slice_Single_Volume(
            pdf_path = Path(pdf_path),
            volume = volume_index,
            write_txt = False,
        )
        all_chunks.extend(chunks)
        volume_counts[volume_index] = len(chunks)

    # 跨卷统一重排 chunk_id，保证合并后主键连续
    for new_id, chunk in enumerate(all_chunks, start = 1):
        chunk["chunk_id"] = new_id

    # 二级切分只接受文件路径输入，先把一级合并结果落盘成临时文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    temp_path = output_path.parent / f"_{output_path.stem}_level1_temp.json"
    with open(temp_path, "w", encoding = "utf-8") as file_obj:
        json.dump(all_chunks, file_obj, ensure_ascii = False, indent = 2)

    # output_name 用 output_path.stem，二级切分会写到 input_path.parent / 同名.json
    # 正好直接落在 output_path 上，不用再搬一次
    Langchain_Recursive_Slice(
        json_dir = str(temp_path),
        output_name = output_path.stem,
        tokenizer = tokenizer,
        chunk_size = chunk_size,
        overlap_tokens = overlap_tokens,
    )
    temp_path.unlink()

    with open(output_path, encoding = "utf-8") as file_obj:
        final_chunks = json.load(file_obj)

    print(f"一级切分完成，{len(pdf_paths)} 卷，共 {len(all_chunks)} chunks")
    print(f"二级切分完成，输出 -> {output_path}，共 {len(final_chunks)} chunks")

    return {
        "output_path": str(output_path),
        "volume_chunk_counts": volume_counts,
        "level1_chunk_count": len(all_chunks),
        "final_chunk_count": len(final_chunks),
    }