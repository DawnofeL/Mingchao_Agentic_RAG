"""通用向量化工具。

本模块提供一个万用函数 `Vectorization`，用于把任意 `list[dict]` 结构的 JSON
按 `how_to_vectorize` 配置做 BGE-M3 向量化，并把向量直接并回原数据。
"""

import json
from pathlib import Path
from typing import Any

import torch
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm


def _select_device() -> str:
    """选择推理设备，优先级 CUDA > Apple MPS > CPU，逐级退行。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_encoder(embedding_model: str, use_fp16: bool, device: str) -> BGEM3FlagModel:
    model_path = Path(embedding_model)

    if model_path.exists() and model_path.is_dir():
        config_path = model_path / "config.json"
        bge_child = model_path / "BAAI_bge-m3"

        if not config_path.exists() and bge_child.exists() and bge_child.is_dir():
            source = str(bge_child)
        else:
            source = str(model_path)
    else:
        source = embedding_model

    return BGEM3FlagModel(source, use_fp16=use_fp16, devices=device)


def _normalize_how_to_vectorize(
    how_to_vectorize: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    if not isinstance(how_to_vectorize, dict) or not how_to_vectorize:
        raise ValueError("how_to_vectorize 必须是非空 dict。")

    normalized: dict[str, dict[str, str]] = {}
    used_outputs: set[str] = set()
    allowed_modes = {"dense", "sparse"}

    for target, config in how_to_vectorize.items():
        if not isinstance(target, str) or not target.strip():
            raise ValueError("how_to_vectorize 的字段名必须是非空字符串。")
        if not isinstance(config, dict):
            raise ValueError(f"字段 '{target}' 的配置必须是 dict。")

        mode = str(config.get("mode", "")).strip().lower()
        output = str(config.get("output", "")).strip()

        if mode not in allowed_modes:
            raise ValueError(
                f"字段 '{target}' 的 mode 必须是 'dense' 或 'sparse'，当前为 '{mode}'。"
            )
        if not output:
            raise ValueError(f"字段 '{target}' 的 output 不能为空。")
        if output in used_outputs:
            raise ValueError(f"输出字段 '{output}' 被重复使用。")

        normalized[target] = {
            "mode": mode,
            "output": output,
        }
        used_outputs.add(output)

    return normalized


def _collect_target_texts(
    data: list[dict[str, Any]],
    target: str,
    max_text_len: int | None,
    strict: bool,
) -> tuple[list[int], list[str], int]:
    row_indices: list[int] = []
    texts: list[str] = []
    skipped = 0

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            if strict:
                raise ValueError(f"第 {i} 行不是 dict。")
            skipped += 1
            continue

        if target not in item:
            if strict:
                raise KeyError(f"第 {i} 行缺少字段 '{target}'。")
            skipped += 1
            continue

        text = str(item.get(target, "")).strip()
        if not text:
            if strict:
                raise ValueError(f"第 {i} 行字段 '{target}' 为空。")
            skipped += 1
            continue

        if max_text_len is not None:
            text = text[:max_text_len]

        row_indices.append(i)
        texts.append(text)

    if not texts:
        raise ValueError(f"字段 '{target}' 没有可向量化的有效文本。")

    return row_indices, texts, skipped


def _encode_texts(
    encoder: BGEM3FlagModel,
    texts: list[str],
    mode: str,
    batch_size: int,
    target: str,
) -> list[list[float]] | list[dict[int, float]]:
    encoded_vectors: list[list[float]] | list[dict[int, float]] = []
    return_dense = mode == "dense"
    return_sparse = mode == "sparse"

    for start in tqdm(range(0, len(texts), batch_size), desc=f"向量化[{target}:{mode}]"):
        batch = texts[start : start + batch_size]
        out = encoder.encode(
            batch,
            return_dense=return_dense,
            return_sparse=return_sparse,
        )

        if return_dense:
            encoded_vectors.extend(out["dense_vecs"].tolist())
        else:
            for sw in out["lexical_weights"]:
                encoded_vectors.append({int(k): float(v) for k, v in sw.items()})

    return encoded_vectors


def Vectorization(
    json_dir: str,
    how_to_vectorize: dict[str, dict[str, str]],
    embedding_model: str = "model/BAAI_bge-m3",
    output_name: str | None = None,
    batch_size: int = 32,
    use_fp16: bool | None = None,
    max_text_len: int | None = None,
    strict: bool = True,
) -> str:
    """对通用 JSON 按配置执行 BGE-M3 向量化。

    Args:
        json_dir: 输入 JSON 路径（相对路径或绝对路径都可）。
        how_to_vectorize: 向量化配置字典。key 是要编码的字段名，value 是
            `{"mode": "dense" | "sparse", "output": "输出字段名"}` 格式的配置。
        embedding_model: BGE-M3 模型本地路径或 HuggingFace model id。
        output_name: 输出文件名；不填时默认 `<输入文件名>_vectorized.json`。
        batch_size: 批量编码大小。
        use_fp16: 是否强制 fp16；None 时按是否有 CUDA 自动选择。
        max_text_len: 文本截断上限（按字符）；None 表示不截断。
        strict: True 遇到坏行直接报错，False 跳过坏行并在最后打印跳过数量。

    Returns:
        输出 JSON 的路径字符串。
    """
    json_path = Path(json_dir)
    if not json_path.is_absolute():
        json_path = (Path.cwd() / json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = [{"name": k, **v} for k, v in data.items()]
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是 list[dict] 或 dict[str, dict]。")

    if batch_size <= 0:
        raise ValueError("batch_size 必须 > 0。")
    if max_text_len is not None and max_text_len <= 0:
        raise ValueError("max_text_len 必须是正整数或 None。")

    normalized_config = _normalize_how_to_vectorize(how_to_vectorize)

    device = _select_device()
    if use_fp16 is None:
        use_fp16 = (device == "cuda")

    encoder = _load_encoder(embedding_model, use_fp16=use_fp16, device=device)
    total_skipped = 0

    for target, config in normalized_config.items():
        mode = config["mode"]
        output = config["output"]
        row_indices, texts, skipped = _collect_target_texts(
            data=data,
            target=target,
            max_text_len=max_text_len,
            strict=strict,
        )
        total_skipped += skipped

        print(
            f"开始编码: {len(texts)} 条 | 设备={device} | "
            f"目标字段={target} | 模式={mode} | 输出字段={output}"
        )

        vectors = _encode_texts(
            encoder=encoder,
            texts=texts,
            mode=mode,
            batch_size=batch_size,
            target=target,
        )

        for idx, vector in zip(row_indices, vectors):
            data[idx][output] = vector

    if output_name is None:
        out_path = json_path.with_name(f"{json_path.stem}_vectorized.json")
    else:
        out_name = output_name if output_name.lower().endswith(".json") else f"{output_name}.json"
        out_path = json_path.with_name(out_name)

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成: {len(data)} 条 -> {out_path}")
    if total_skipped:
        print(f"累计跳过行数: {total_skipped}")
    return str(out_path)