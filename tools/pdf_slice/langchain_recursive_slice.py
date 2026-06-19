"""二级切分工具：对一级切分结果按 token 阈值递归切分。

本文件定义 `Langchain_Recursive_Slice` 主函数，输入一级切分 JSON，
输出二级切分 JSON。切分流程基于 `RecursiveCharacterTextSplitter`，
并在后处理阶段追加句子级 overlap，保证 overlap 边界尽量落在句尾。
"""

import json
import re
from pathlib import Path

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


def Langchain_Recursive_Slice(
    json_dir: str,
    output_name: str,
    tokenizer,
    chunk_size: int = 1000,
    overlap_tokens: int = 50,
    separators: list = None,
    strip_whitespace: bool = True,
):
    """对输入 chunk JSON 做二次分割，超过 chunk_size token 的 chunk 递归切分。

    切分策略：
    - 主切点严格对齐句尾（。！？……），不在逗号等非句尾处截断
    - LangChain 内置 overlap 不走分隔符逻辑，会强切句子，故关掉（chunk_overlap = 0）
    - 转而用 Add_Sentence_Overlap 后处理：从前一个 chunk 末尾取若干完整句子
      作为 overlap 前缀，直到凑满 overlap_tokens，保证 overlap 边界也对齐句尾

    Args:
        json_dir: 输入 JSON 文件路径，绝对或相对路径均可。
        output_name: 输出文件名（含或不含 .json 均可）。
        tokenizer: tokenizer 对象，或本地 tokenizer 目录路径（str/Path）。
        chunk_size: 每块目标 token 数，默认 1000。
        overlap_tokens: overlap 的 token 预算（从前一 chunk 末尾取完整句子），默认 50。
        separators: 分隔符优先级列表，None 时用中文标点默认值。
        strip_whitespace: 是否去掉每块首尾空白，默认 True。
    Returns:
        list[dict]，每条含 chunk_id/volume/chapter/section/chunk_index/chunk_total/content。
    """

    # 这里把 tokenizer 的两种输入形态统一起来，避免调用方必须关心加载细节。
    def Load_Tokenizer(tokenizer_like):
        """加载 tokenizer；支持直接传对象或本地路径。"""
        if not isinstance(tokenizer_like, (str, Path)) and hasattr(tokenizer_like, "encode"):
            return tokenizer_like, "tokenizer_object"

        # 相对路径统一转绝对路径，避免 notebook 切换工作目录后找错模型。
        tokenizer_path = Path(tokenizer_like)
        if not tokenizer_path.is_absolute():
            tokenizer_path = (Path.cwd() / tokenizer_path).resolve()
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer 路径不存在: {tokenizer_path}")

        from transformers import AutoTokenizer

        tokenizer_obj = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code = True,
            local_files_only = True,
        )
        return tokenizer_obj, str(tokenizer_path)

    # 这里专门做一层兼容，减少不同 transformers 版本带来的 encode 接口差异。
    def Build_Token_Len(tokenizer_obj):
        """构造兼容不同 transformers 版本的 token 计数函数。"""

        def Token_Len(text: str) -> int:
            content_text = str(text)
            try:
                return len(tokenizer_obj.encode(content_text, add_special_tokens = False))
            except TypeError:
                return len(tokenizer_obj.encode(content_text))
            except Exception:
                return len(tokenizer_obj(content_text, add_special_tokens = False)["input_ids"])

        return Token_Len

    # overlap 不直接用 LangChain 的 chunk_overlap，而是后处理句子级拼接。
    # 这样可以避免"在句中硬切"导致的检索片段语义断裂。
    def Add_Sentence_Overlap(pieces: list[str], token_len_fn, overlap_token_count: int) -> list[str]:
        """给切分后的 pieces 添加句子感知 overlap。"""
        # sentence_pattern 会尽量让分句落在句尾标点，末尾无标点也兜底保留。
        sentence_pattern = re.compile(r"[^。！？…]+[。！？…]+|[^。！？…]+$")

        merged_pieces = [pieces[0]]
        for piece_index in range(1, len(pieces)):
            previous_sentences = sentence_pattern.findall(pieces[piece_index - 1])
            overlap_sentences = []
            used_tokens = 0

            for sentence in reversed(previous_sentences):
                sentence_cost = token_len_fn(sentence)
                if used_tokens + sentence_cost <= overlap_token_count:
                    overlap_sentences.insert(0, sentence)
                    used_tokens += sentence_cost
                else:
                    break

            if overlap_sentences:
                merged_pieces.append("".join(overlap_sentences) + pieces[piece_index])
            else:
                merged_pieces.append(pieces[piece_index])

        return merged_pieces

    # 统一输入路径解析规则：相对路径按当前工作目录解释。
    input_path = Path(json_dir)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 输出名允许不带后缀，这里统一补成 .json，保证行为稳定。
    output_file_name = output_name if str(output_name).lower().endswith(".json") else f"{output_name}.json"
    output_path = input_path.parent / output_file_name

    # 输入必须是 list，后续流程默认逐条 dict 处理。
    with open(input_path, "r", encoding = "utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 必须是 list 格式")

    # tokenizer 与 token 计数函数在入口只构建一次，循环内复用。
    tokenizer_obj, tokenizer_source = Load_Tokenizer(tokenizer)
    token_len = Build_Token_Len(tokenizer_obj)

    # 默认分隔符按"段落 -> 换行 -> 句尾标点 -> 空格 -> 兜底字符切分"的顺序。
    default_separators = ["\n\n", "\n", "。", "！", "？", "……", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        separators = separators if separators is not None else default_separators,
        chunk_size = chunk_size,
        chunk_overlap = 0,
        length_function = token_len,
        is_separator_regex = False,
        keep_separator = "end",
        strip_whitespace = strip_whitespace,
    )

    output_rows = []
    next_chunk_id = 1
    kept_count = 0
    split_parent_count = 0

    # 主循环只关注"有效 dict 且 content 非空"的记录，坏数据直接跳过。
    for item in tqdm(data, desc = "二级切分"):
        if not isinstance(item, dict):
            continue

        content = str(item.get("content", "")).strip()
        if not content:
            continue

        volume = item.get("volume")
        chapter = item.get("chapter")
        section = item.get("section")

        # 小于阈值直接保留，保证短文本不被过度切分。
        token_count = token_len(content)
        if token_count <= chunk_size:
            piece_list = [content]
            kept_count += 1
        else:
            # 先做递归切分，再做句子级 overlap；两步分开可读性更高、排错更直观。
            raw_pieces = [part.strip() for part in splitter.split_text(content) if str(part).strip()]
            if not raw_pieces:
                raw_pieces = [content]
            piece_list = Add_Sentence_Overlap(raw_pieces, token_len, overlap_tokens)
            split_parent_count += 1

        # 每个父 chunk 内部维护 chunk_index/chunk_total，全局维护连续 chunk_id。
        chunk_total = len(piece_list)
        for chunk_index, piece in enumerate(piece_list, start = 1):
            output_rows.append({
                "chunk_id": next_chunk_id,
                "volume": volume,
                "chapter": chapter,
                "section": section,
                "chunk_index": chunk_index,
                "chunk_total": chunk_total,
                "content": piece,
            })
            next_chunk_id += 1

    # 输出采用 UTF-8 且带缩进，便于人工检查与后续版本对比。
    with open(output_path, "w", encoding = "utf-8") as file_obj:
        json.dump(output_rows, file_obj, ensure_ascii = False, indent = 2)

    # 统计信息用于快速判断切分行为是否符合预期，不必额外打开结果文件。
    print(f"tokenizer:     {tokenizer_source}")
    print(f"chunk_size:    {chunk_size}  overlap_tokens: {overlap_tokens}")
    print(f"in:  {input_path}")
    print(f"out: {output_path}")
    print(f"input chunks: {len(data)} | kept<={chunk_size}: {kept_count} | split>{chunk_size}: {split_parent_count}")
    print(f"output chunks: {len(output_rows)}")
    return output_rows