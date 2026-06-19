"""Milvus 通用 JSON 插入工具。

适用于 Docker 部署的完整 Milvus（非 Lite）。
`Insert_json_Into_Milvus_Collection` 是主入口：若数据库或 collection 不存在则自动创建，
然后批量插入 JSON 数据并返回统计信息。
"""

import json
from pathlib import Path

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, db, utility


_DATA_TYPE_MAP = {
    "Int8":              DataType.INT8,
    "Int16":             DataType.INT16,
    "Int32":             DataType.INT32,
    "Int64":             DataType.INT64,
    "Float":             DataType.FLOAT,
    "Double":            DataType.DOUBLE,
    "VarChar":           DataType.VARCHAR,
    "String":            DataType.VARCHAR,
    "Bool":              DataType.BOOL,
    "FloatVector":       DataType.FLOAT_VECTOR,
    "BinaryVector":      DataType.BINARY_VECTOR,
    "SparseFloatVector": DataType.SPARSE_FLOAT_VECTOR,
}


def _get_type_param(field_def: dict, key: str) -> str | None:
    for param in field_def.get("type_params", []):
        if param.get("key") == key:
            return param.get("value")
    return None


def _Build_Collection_From_Schema(collection_name: str, schema_json_path: str) -> Collection:
    """从 Attu 导出的 schema JSON 建 collection，并自动推断向量索引。"""
    with open(schema_json_path, "r", encoding="utf-8") as f:
        schema_data = json.load(f)

    raw_fields = schema_data["schema"]["fields"]
    field_schemas = []

    for field_def in raw_fields:
        dtype_str = field_def["data_type"]
        dtype = _DATA_TYPE_MAP.get(dtype_str)
        if dtype is None:
            raise ValueError(f"不支持的字段类型: {dtype_str}（字段 {field_def['name']}）")

        kwargs = {}
        if field_def.get("is_primary_key"):
            kwargs["is_primary"] = True
        if field_def.get("nullable"):
            kwargs["nullable"] = True

        if dtype == DataType.VARCHAR:
            max_len = _get_type_param(field_def, "max_length")
            kwargs["max_length"] = int(max_len) if max_len else 65535

        if dtype == DataType.FLOAT_VECTOR:
            dim = _get_type_param(field_def, "dim")
            if not dim:
                raise ValueError(f"FloatVector 字段 '{field_def['name']}' 缺少 dim。")
            kwargs["dim"] = int(dim)

        field_schemas.append(FieldSchema(name=field_def["name"], dtype=dtype, **kwargs))

    schema = CollectionSchema(fields=field_schemas, description="")
    collection = Collection(name=collection_name, schema=schema)

    # 向量字段自动推断索引：dense 用 AUTOINDEX/COSINE，sparse 用 SPARSE_INVERTED_INDEX/IP
    # Attu 导出的 index_params 始终为空，因此这里按 data_type 自行推断
    for field_def in raw_fields:
        dtype_str = field_def["data_type"]
        if dtype_str == "FloatVector":
            collection.create_index(
                field_name   = field_def["name"],
                index_params = {"index_type": "AUTOINDEX", "metric_type": "COSINE"},
            )
        elif dtype_str == "SparseFloatVector":
            collection.create_index(
                field_name   = field_def["name"],
                index_params = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
            )

    print(f"[milvus] collection '{collection_name}' 创建完成")
    return collection


def Resolve_Json_Source(field: list[str], json_correspond) -> tuple[Path, dict[str, str]]:
    """解析 json_correspond，得到 JSON 路径和字段映射。"""
    # 简写模式：直接给 JSON 路径时，默认按"同名字段"做映射。
    if isinstance(json_correspond, (str, Path)):
        json_path = Path(json_correspond).resolve()
        field_map = {name: name for name in field}
        return json_path, field_map

    # 高级模式：允许显式指定 json 路径与字段映射，适合异名字段场景。
    if isinstance(json_correspond, dict):
        json_dir = json_correspond.get("json_dir", json_correspond.get("json_path", ""))
        if not json_dir:
            raise ValueError("json_correspond 缺少 json_dir 或 json_path。")

        raw_map = json_correspond.get("field_map", json_correspond.get("map", {}))
        if not raw_map:
            field_map = {name: name for name in field}
        else:
            field_map = {str(key): str(value) for key, value in raw_map.items()}

        return Path(str(json_dir)).resolve(), field_map

    raise TypeError("json_correspond 仅支持 str/Path 或 dict。")


def Build_Field_Type_Map(collection: Collection) -> dict[str, int]:
    """从 collection schema 构建字段类型映射。"""
    type_map = {}

    # 预先缓存 schema 字段类型，后续逐行转换时避免重复查 schema。
    for schema_field in collection.schema.fields:
        type_map[schema_field.name] = schema_field.dtype

    return type_map


def Convert_Value_By_Type(value, dtype: int):
    """按 Milvus 字段类型转换值。"""
    # 标量字段做基础强转，优先把类型问题前置暴露。
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
        return int(value)

    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return float(value)

    if dtype in (DataType.VARCHAR, DataType.STRING):
        return str(value)

    if dtype == DataType.BOOL:
        return bool(value)

    # 稠密向量必须是 list，且每个元素强转 float，避免 numpy/object 混入。
    if dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
        if not isinstance(value, list):
            raise TypeError("向量字段必须是 list。")
        return [float(number) for number in value]

    # 稀疏向量要求 dict[token_id -> weight]，这里统一转成 int/float。
    if dtype == DataType.SPARSE_FLOAT_VECTOR:
        if not isinstance(value, dict):
            raise TypeError("稀疏向量字段必须是 dict。")
        return {int(key): float(number) for key, number in value.items()}

    return value


def Build_Insert_Rows(
    rows: list[dict],
    field: list[str],
    field_map: dict[str, str],
    field_type_map: dict[str, int],
) -> tuple[list[dict], list[dict]]:
    """构建插入行并收集错误信息。"""
    insert_rows = []
    failed_rows = []

    # 逐行校验并构造 insert_row，保证每一行要么完整入库要么完整失败。
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            failed_rows.append({"row_index": row_index, "reason": "行不是 dict"})
            continue

        insert_row = {}
        row_ok = True

        # 按目标 collection 字段顺序取值并转换，避免漏列或错列。
        for field_name in field:
            json_key = field_map.get(field_name, field_name)

            if field_name not in field_type_map:
                row_ok = False
                failed_rows.append(
                    {"row_index": row_index, "reason": f"collection 缺少字段 {field_name}"}
                )
                break

            if json_key not in row:
                row_ok = False
                failed_rows.append(
                    {"row_index": row_index, "reason": f"JSON 缺少字段 {json_key}"}
                )
                break

            try:
                insert_row[field_name] = Convert_Value_By_Type(
                    value = row[json_key],
                    dtype = field_type_map[field_name],
                )
            except Exception as error:
                row_ok = False
                failed_rows.append(
                    {
                        "row_index": row_index,
                        "reason": f"字段 {field_name} 类型转换失败: {error}",
                    }
                )
                break

        # 只把"整行成功"的记录加入插入队列，失败行保留错误样本供排查。
        if row_ok:
            insert_rows.append(insert_row)

    return insert_rows, failed_rows


def Insert_json_Into_Milvus_Collection(
    milvus_database: str,
    milvus_collection: str,
    field: list[str],
    json_correspond,
    schema_json: str | None = None,
    batch_size: int = 200,
    host: str = "localhost",
    port: str = "19530",
    strict: bool = True,
) -> dict:
    """把任意 JSON 按字段映射批量插入任意 Milvus collection。

    若数据库不存在则自动创建；若 collection 不存在且提供了 schema_json 则自动建 collection。

    Args:
        milvus_database: 目标 Milvus 数据库名。
        milvus_collection: 目标 collection 名。
        field: 需要写入的 collection 字段列表。
        json_correspond: JSON 来源与字段映射。简写：直接传路径字符串，默认同名映射。
            完整写法：{"json_dir": "xxx.json", "field_map": {"collection字段": "json字段"}}。
        schema_json: Attu 导出的 collection describe JSON 路径。collection 不存在时用于自动建表。
        batch_size: 每批插入条数，默认 200。
        host: Milvus host，默认 localhost。
        port: Milvus port，默认 19530。
        strict: True 时任意失败行直接报错，False 时跳过失败行继续插入。

    Returns:
        包含插入统计和失败样本的字典。
    """
    if not field:
        raise ValueError("field 不能为空。")

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")

    # 解析 json 源与字段映射，支持"路径直传"和"映射字典"两种模式。
    json_path, field_map = Resolve_Json_Source(field=field, json_correspond=json_correspond)

    if not json_path.exists():
        raise FileNotFoundError(f"找不到 JSON 文件: {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list):
        raise ValueError("输入 JSON 顶层必须是 list。")

    # 先连接默认库，确保连接建立后再操作数据库和 collection。
    connections.connect(alias="default", host=host, port=port)

    # 没有就建库，有就直接切过去。
    existing_dbs = db.list_database()
    if milvus_database not in existing_dbs:
        db.create_database(milvus_database)
        print(f"[milvus] 数据库 '{milvus_database}' 创建完成")
    else:
        print(f"[milvus] 数据库 '{milvus_database}' 已存在")
    db.using_database(milvus_database)

    # 没有 collection 就从 schema_json 建，有就直接用。
    if not utility.has_collection(milvus_collection):
        if schema_json is None:
            raise ValueError(
                f"collection '{milvus_collection}' 不存在，请提供 schema_json 路径以自动建 collection。"
            )
        schema_path = Path(schema_json).resolve()
        if not schema_path.exists():
            raise FileNotFoundError(f"schema_json 不存在: {schema_path}")
        collection = _Build_Collection_From_Schema(milvus_collection, str(schema_path))
    else:
        collection = Collection(milvus_collection)
        print(f"[milvus] collection '{milvus_collection}' 已存在，直接插入")

    field_type_map = Build_Field_Type_Map(collection=collection)

    # 在写入前先校验 collection 是否具备目标字段，防止"跑到一半失败"。
    missing_fields = [name for name in field if name not in field_type_map]
    if missing_fields:
        raise ValueError(f"collection 缺少字段: {missing_fields}")

    insert_rows, failed_rows = Build_Insert_Rows(
        rows           = rows,
        field          = field,
        field_map      = field_map,
        field_type_map = field_type_map,
    )

    # strict 模式下只要出现失败行就终止，避免"部分脏写"。
    if strict and failed_rows:
        preview = failed_rows[:5]
        raise ValueError(f"发现失败行 {len(failed_rows)} 条，示例: {preview}")

    # 分批写入控制内存峰值与网络负载，flush 后再取实体总数。
    for start in range(0, len(insert_rows), batch_size):
        collection.insert(insert_rows[start : start + batch_size])

    collection.flush()

    result = {
        "json_path":           str(json_path),
        "database":            milvus_database,
        "collection":          milvus_collection,
        "total_rows":          len(rows),
        "inserted_rows":       len(insert_rows),
        "failed_rows":         len(failed_rows),
        "failed_examples":     failed_rows[:10],
        "collection_entities": collection.num_entities,
    }
    print(result)
    return result