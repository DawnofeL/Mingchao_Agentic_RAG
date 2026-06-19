"""Milvus Lite 首次运行初始化。

仅在 MILVUS_MODE = "lite" 时调用。
检查本地 .db 文件中 collection 是否已存在，不存在则从预计算好的
JSON（chunk 1-661）导入数据，无需重跑 embedding 模型。
"""

import json
from pathlib import Path

from pymilvus import DataType, MilvusClient

from rag.config.settings import (
    CHUNK_COLLECTION,
    MILVUS_LITE_CHUNKS_JSON,
    MILVUS_LITE_PATH,
)


def ensure_lite_db() -> None:
    """确保 Milvus Lite DB 中存在 chunk collection，不存在则自动导入。"""

    client = MilvusClient(uri=MILVUS_LITE_PATH)

    if CHUNK_COLLECTION in client.list_collections():
        print(f"[lite_setup] Collection '{CHUNK_COLLECTION}' 已存在，跳过导入")
        return

    print(f"[lite_setup] 首次使用 Lite 模式，开始建表并导入 chunk 数据...")

    # 建 schema（与 Docker Milvus 中 Ming_Dynasty 的 schema 保持一致）
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id",        DataType.INT64,          is_primary=True)
    schema.add_field("volume",          DataType.INT64)
    schema.add_field("chapter",         DataType.VARCHAR,        max_length=64)
    schema.add_field("section",         DataType.VARCHAR,        max_length=256)
    schema.add_field("chunk_index",     DataType.INT64)
    schema.add_field("chunk_total",     DataType.INT64)
    schema.add_field("content",         DataType.VARCHAR,        max_length=16384)
    schema.add_field("embedding",       DataType.FLOAT_VECTOR,   dim=1024)
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

    # 建索引（与 Attu 截图中的配置保持一致）
    index_params = client.prepare_index_params()
    index_params.add_index("embedding",        index_type="AUTOINDEX",            metric_type="COSINE")
    # sparse 用 AUTOINDEX 在 Lite 模式下落盘后重启 load 会崩（binary 格式不被索引构建器认）
    # 必须显式指定 SPARSE_INVERTED_INDEX，跨进程 reload 才正常
    index_params.add_index("sparse_embedding", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

    client.create_collection(
        collection_name=CHUNK_COLLECTION,
        schema=schema,
        index_params=index_params,
    )

    # 读取预计算 JSON，转换 sparse key：str → int（Milvus 要求 int 键）
    raw = json.loads(Path(MILVUS_LITE_CHUNKS_JSON).read_text(encoding="utf-8"))
    rows = []
    for d in raw:
        row = dict(d)
        row["sparse_embedding"] = {int(k): float(v) for k, v in d["sparse_embedding"].items()}
        rows.append(row)

    client.insert(collection_name=CHUNK_COLLECTION, data=rows)
    print(f"[lite_setup] 导入完成，共 {len(rows)} 条")
