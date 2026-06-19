"""明朝 PDF 分卷切分与合并工具。

本文件定义的函数与职责如下。
`Strip_Repeats` 用于清理 PDF 叠印造成的重复字符串片段。
`Dedup_Chapter_Title` 用于修复章标题的叠印噪声并返回有效标题长度。
`Extract_Tail_Number` 用于从文件名末尾提取排序数字。
`Resolve_Output_Dir` 用于根据输入目录推断输出目录。
`Slice_Single_Volume` 用于切分单个分卷 PDF 并返回该卷 chunks。
`Merge_Volume_Json` 用于按尾号顺序合并单卷 JSON 并重排 chunk_id。
`Ming_Volume_Slice` 是主入口，负责批量扫描 PDF、逐卷输出 JSON、最终生成总文件。
"""

import json
import re
from pathlib import Path

import pypdf


def Strip_Repeats(text: str, min_repeat: int = 3) -> tuple[str, int]:
    """剥离字符串开头重复片段并返回消耗长度。

    Args:
        text: 待处理原文。
        min_repeat: 至少重复次数阈值。
    Returns:
        (clean_text, consumed_len)。
    """
    # 从最短重复单元开始尝试，优先消除"叠印复制"而不是误伤正文。
    for unit_len in range(1, len(text) // min_repeat + 1):
        unit = text[:unit_len]
        ptr = 0
        count = 0

        # 连续统计当前单元重复次数，找到第一个满足阈值的重复块。
        while text[ptr:].startswith(unit):
            ptr += unit_len
            count += 1

        if count < min_repeat:
            continue

        # 把重复段后面的尾部继续递归处理，逐层剥离叠印噪声。
        tail = text[ptr:]

        if not tail:
            return unit, ptr

        tail_clean, tail_consumed = Strip_Repeats(tail, min_repeat = min_repeat)

        if tail_consumed == 0:
            return unit, ptr

        return unit + tail_clean, ptr + tail_consumed

    return "", 0


def Dedup_Chapter_Title(raw_title: str) -> tuple[str, int]:
    """去掉章标题叠印噪声并返回标题占用字符数。

    Args:
        raw_title: 章标题原始串。
    Returns:
        (clean_title, consumed_len)。
    """
    # 先锁定"第X章"前缀，后续只清洗前缀后的叠印内容。
    match = re.match(r"(第[一二三四五六七八九十百零]+章)", raw_title)

    if not match:
        return raw_title.strip(), len(raw_title)

    prefix = match.group(1)
    ptr = match.end()

    while ptr < len(raw_title) and raw_title[ptr] in " \t　":
        ptr += 1

    while raw_title[ptr:].startswith(prefix):
        ptr += len(prefix)

        # 章号重复后可能夹杂全角空格，这里一并吞掉。
        while ptr < len(raw_title) and raw_title[ptr] in " \t　":
            ptr += 1

    rest = raw_title[ptr:]

    if not rest:
        return prefix, ptr

    title_body, consumed = Strip_Repeats(rest)

    if consumed == 0:
        return f"{prefix} {rest}".strip(), len(raw_title)

    # consumed 用于上层精确计算正文起点，避免标题与正文粘连。
    return f"{prefix} {title_body}".strip(), ptr + consumed


def Extract_Tail_Number(file_stem: str) -> int:
    """提取文件名最后一个数字用于排序。"""
    nums = re.findall(r"\d+", file_stem)

    if not nums:
        return 10 ** 9

    return int(nums[-1])


def Resolve_Output_Dir(pdf_root: Path) -> Path:
    """推断输出目录。

    输入目录名以 `_pdf` 结尾时，输出到同级 `_json` 目录。
    其他情况输出到输入目录本身。
    """
    # 约定式目录映射，减少调用方手动传 output_dir 的心智负担。
    if pdf_root.name.endswith("_pdf"):
        return pdf_root.parent / f"{pdf_root.name[:-4]}_json"

    return pdf_root


def Slice_Single_Volume(pdf_path: Path, volume: int, write_txt: bool = False) -> list[dict]:
    """按旧版 Volume_Slice 规则切分单卷 PDF。"""
    raw_pages = []

    # 逐页提取文本并做基础清洗，先把可见噪声尽量过滤掉。
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)

        for page in reader.pages:
            text = page.extract_text() or ""
            lines = text.splitlines()

            # 页码常表现为独立全角数字行，先去掉避免污染后续分章。
            lines = [line for line in lines if not re.match(r"^\s*[０-９]+\s*$", line)]
            cleaned = []
            buf = ""

            for line in lines:
                line = line.strip()

                if not line:
                    if buf:
                        cleaned.append(buf)
                        buf = ""

                    continue

                # 短行常是被 PDF 拆断的半句，先拼到缓冲区再判断是否落盘。
                if len(line) <= 2:
                    buf += line
                else:
                    buf += line
                    cleaned.append(buf)
                    buf = ""

            if buf:
                cleaned.append(buf)

            raw_pages.append("\n".join(cleaned))

    full_text = "\n".join(raw_pages)
    full_text = re.sub(r"\n[０-９]+", "", full_text)

    # 可选导出纯文本，便于肉眼排查切分边界是否异常。
    if write_txt:
        txt_path = pdf_path.with_suffix(".txt")
        txt_path.write_text(full_text, encoding = "utf-8")
        print(f"[txt] {txt_path}")

    # chapter_pat 找章标题，sub_pat 找【小节】；统一转事件流后做时序切分。
    chapter_pat = re.compile(r"第([一二三四五六七八九十百零]+)章[\s　]*([^\n【]*)")
    sub_pat = re.compile(r"【([^】]+)】")
    events = []

    for match in chapter_pat.finditer(full_text):
        clean_title, consumed = Dedup_Chapter_Title(match.group(0))
        events.append((match.start(), "chapter", clean_title, match.start() + consumed))

    for match in sub_pat.finditer(full_text):
        events.append((match.start(), "sub", match.group(1).strip(), match.end()))

    # 事件按全文位置排序，后续可以单次线性扫描完成所有切块。
    events.sort(key = lambda item: item[0])
    chunks = []
    current_chapter = ""
    current_chapter_idx = 0
    pending_start = None

    def Make_Chunk(chapter: str, section: str, text: str) -> dict | None:
        # 统一清理多余空行，避免 chunk 内容因版式噪声看起来"断裂"。
        content = re.sub(r"\n{2,}", "\n", text).strip()
        cn_count = sum(1 for ch in content if "一" <= ch <= "鿿")

        # 过短中文片段通常是残留噪声，直接丢弃可提高 chunk 纯度。
        if cn_count < 10:
            return None

        return {
            "chunk_id": len(chunks) + 1,
            "volume": volume,
            "chapter": chapter,
            "section": section,
            "content": content,
        }

    def Flush_Intro(up_to_pos: int) -> None:
        # 引言只在"章标题之后且尚未进入小节"这段范围内尝试生成。
        if pending_start is None or not current_chapter:
            return

        intro = re.sub(r"\s+", " ", full_text[pending_start:up_to_pos]).strip()
        cn_count = sum(1 for ch in intro if "一" <= ch <= "鿿")

        # 引言太短通常是排版碎片，不单独成块可减少噪声召回。
        if cn_count <= 20:
            return

        intro_chunk = Make_Chunk(
            chapter = current_chapter,
            section = f"【第{current_chapter_idx}章引言】",
            text = intro,
        )

        if intro_chunk:
            chunks.append(intro_chunk)

    idx = 0

    # 线性扫 events：遇章切章，遇小节切小节，同时在边界冲刷引言段。
    while idx < len(events):
        pos, event_type = events[idx][0], events[idx][1]

        if event_type == "chapter":
            Flush_Intro(up_to_pos = pos)
            current_chapter = events[idx][2]
            current_chapter_idx += 1
            pending_start = events[idx][3]
            idx += 1
            continue

        if event_type == "sub":
            section_title = f"【{events[idx][2]}】"

            # 小节开始前先冲刷一次引言，保证引言不会吞进第一个小节块。
            Flush_Intro(up_to_pos = pos)
            next_pos = events[idx + 1][0] if idx + 1 < len(events) else len(full_text)
            sub_chunk = Make_Chunk(
                chapter = current_chapter,
                section = section_title,
                text = full_text[pos:next_pos],
            )

            if sub_chunk:
                chunks.append(sub_chunk)

            pending_start = next_pos
            idx += 1
            continue

        idx += 1

    # 处理最后一个事件后的尾部，引言补齐在这里完成。
    Flush_Intro(up_to_pos = len(full_text))
    return chunks


def Merge_Volume_Json(
    output_dir: Path,
    json_paths: list[Path] | None = None,
    merged_name: str = "明朝那些事儿_chunks.json",
) -> Path:
    """按文件尾号升序合并单卷 JSON 并重排 chunk_id。"""
    if json_paths is None:
        json_files = []

        # 默认扫描输出目录下所有 json，排除最终合并文件自身。
        for json_path in output_dir.glob("*.json"):
            if json_path.name == merged_name:
                continue

            json_files.append(json_path)
    else:
        json_files = [path for path in json_paths if path.name != merged_name]

    json_files = sorted(
        json_files,
        key = lambda path: (Extract_Tail_Number(path.stem), path.name),
    )

    if not json_files:
        raise FileNotFoundError(f"未找到可合并的 JSON 文件: {output_dir}")

    merged = []
    next_chunk_id = 1

    # 合并时统一重排 chunk_id，保证跨卷后主键连续。
    for json_path in json_files:
        with open(json_path, "r", encoding = "utf-8") as file:
            data = json.load(file)

        # 保护性跳过坏文件，避免一个文件损坏导致整批流程终止。
        if not isinstance(data, list):
            print(f"[merge-skip] {json_path.name} 不是 list，已跳过")
            continue

        for item in data:
            if not isinstance(item, dict):
                continue

            # 仅保留统一 schema 字段，避免不同源文件扩展字段相互污染。
            merged.append(
                {
                    "chunk_id": next_chunk_id,
                    "volume": item.get("volume"),
                    "chapter": item.get("chapter"),
                    "section": item.get("section"),
                    "content": item.get("content", ""),
                }
            )
            next_chunk_id += 1

        print(f"[merge] {json_path.name}: {len(data)} chunks")

    merged_path = output_dir / merged_name

    # 最终合并文件同样使用 UTF-8 + 缩进，便于审阅与版本管理。
    with open(merged_path, "w", encoding = "utf-8") as file:
        json.dump(merged, file, ensure_ascii = False, indent = 2)

    print(f"\n输出 -> {merged_path}  共 {len(merged)} chunks")
    return merged_path


def Ming_Volume_Slice(pdf_dir: str) -> dict:
    """批量切分分卷 PDF，并自动生成总合并 JSON。

    Args:
        pdf_dir: 存放多个分卷 PDF 的目录路径。
    Returns:
        包含输出目录、单卷输出列表、总文件路径与统计信息的字典。
    """
    # 先规范并校验输入目录，尽早给出可定位错误。
    pdf_root = Path(pdf_dir).resolve()

    if not pdf_root.exists() or not pdf_root.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {pdf_root}")

    # 输出目录按约定自动推断，保持调用接口最小化。
    output_dir = Resolve_Output_Dir(pdf_root = pdf_root)
    output_dir.mkdir(parents = True, exist_ok = True)

    # 文件处理顺序固定为"尾号升序"，保证结果稳定可复现。
    pdf_files = sorted(
        list(pdf_root.glob("*.pdf")),
        key = lambda path: (Extract_Tail_Number(path.stem), path.name),
    )

    if not pdf_files:
        raise FileNotFoundError(f"目录中未找到 PDF 文件: {pdf_root}")

    single_outputs = []
    generated_paths = []
    total_chunks = 0

    # 逐卷切分并落盘，随后统一进入合并流程。
    for pdf_path in pdf_files:
        volume = Extract_Tail_Number(pdf_path.stem)

        # 非标准命名（无尾号）兜底设为 0，防止中断整批任务。
        if volume >= 10 ** 9:
            volume = 0

        chunks = Slice_Single_Volume(
            pdf_path = pdf_path,
            volume = volume,
            write_txt = False,
        )
        out_json_path = output_dir / f"{pdf_path.stem}.json"

        with open(out_json_path, "w", encoding = "utf-8") as file:
            json.dump(chunks, file, ensure_ascii = False, indent = 2)

        single_outputs.append(str(out_json_path))
        generated_paths.append(out_json_path)
        total_chunks += len(chunks)
        print(f"[slice] {pdf_path.name} -> {out_json_path.name} ({len(chunks)} chunks)")

    # 用本轮实际生成的文件列表做合并，避免误混入历史残留文件。
    merged_path = Merge_Volume_Json(output_dir = output_dir, json_paths = generated_paths)

    return {
        "pdf_dir": str(pdf_root),
        "output_dir": str(output_dir),
        "single_json_files": single_outputs,
        "merged_json_file": str(merged_path),
        "pdf_count": len(pdf_files),
        "total_chunks": total_chunks,
    }