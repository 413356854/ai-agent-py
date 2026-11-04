# =============================================================================
# rag_chunk —— 公司税法 RAG 的检索块表（Milvus 2.5+，推荐 2.6）
#
# 一行 = 一个检索块。**文档级属性不进 Milvus**：文件名、后缀、页数、创建时间、
# 入库统计都在 MySQL `tb_document_parse_job` 里，`doc_id` 就是该表的 `id`，
# 检索后按 doc_id 回查即可拼出引用，不需要在每个块上重复存一份。
#
# 字段（7 个，其中 2 个向量）：
#   身份   chunk_id(PK) / doc_id / chunk_index / page
#   正文   content（同时是 BM25 输入）
#   向量   embedding（稠密） / sparse（BM25 稀疏，服务端生成）
#
# 相比早期的 24 字段版，砍掉了三类：
#   * 文档级、每行重复：source / source_type / title / lang / doc_type / tags /
#     version / created_at / updated_at —— MySQL 已有，且改标题不该更新 N 行；
#   * 当前链路没有生产者：parent_id / section / token_count / metadata
#     （content_hash 的信息已包含在 chunk_id = sha1(doc_id|chunk_index|content) 里）；
#   * 单租户场景下的常量：tenant_id / kb_id。将来要加，普通列用 2.6 的 add_field，
#     分区键则必须重建集合（当前数据量下秒级）。
#
# 关键约束（Milvus 官方文档 + 2.6 实测）：
#   * VARCHAR.max_length 单位是 **UTF-8 字节**，上限 65535（中文 1 字≈3 字节）；
#   * BM25 需要三件套：文本字段 enable_analyzer + SPARSE_FLOAT_VECTOR 输出字段 +
#     FunctionType.BM25 内置函数；sparse 由服务端写，插入时不要赋值；
#   * 标量索引目前只有 INVERTED 可用；ARRAY 字段不支持标量索引；
#   * analyzer 是建表时固定的，换分词/换 embedding 模型都要新建集合再回填；
#   * 集合在 Milvus 重启后是 released 状态，检索前要 load_collection。
#
# 执行：
#   uv run python docs/milvus/create_database.py                    # 先建库 db_tax_rag
#   uv run python docs/milvus/create_rag_chunk.py --dim 1536        # 默认建到 db_tax_rag
#   uv run python docs/milvus/create_rag_chunk.py --recreate        # 重建（会删数据）
#   uv run python docs/milvus/create_rag_chunk.py --print-spec      # 不连库，只打印字段/索引
#
# 说明：Milvus 没有 SQL DDL，建表即调用 pymilvus；--print-spec 输出的 JSON 可给
# REST / 其他语言团队复用。
# =============================================================================
"""创建 rag_chunk 集合（7 字段 + 3 索引 + BM25 函数）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

DEFAULT_DATABASE = "db_tax_rag"
DEFAULT_COLLECTION = "rag_chunk"
DEFAULT_DIM = 1536
DEFAULT_METRIC = "COSINE"
DEFAULT_ANALYZER = "chinese"
DEFAULT_CONSISTENCY = "Bounded"
MAX_CONTENT_BYTES = 65_535

# 用「显式 tokenizer + filter」而不是 {"type": "chinese"} 简写：官方 2.5/2.6 文档
# 都给出展开形式，且部分发行版（Milvus Lite 2.6）只认 tokenizer 名。
ANALYZER_PARAMS: dict[str, dict[str, Any]] = {
    "chinese": {"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
    "english": {
        "tokenizer": "standard",
        "filter": [
            "lowercase",
            {"type": "stemmer", "language": "english"},
            {"type": "stop", "stop_words": "_english_"},
        ],
    },
    "standard": {"tokenizer": "standard"},
}

# (字段名, 数据类型, 参数, 说明)
FIELD_SPECS: tuple[tuple[str, str, dict[str, Any], str], ...] = (
    (
        "chunk_id",
        "VARCHAR",
        {"is_primary": True, "max_length": 64},
        "主键：chunk- + sha1(doc_id|chunk_index|content)[:32]，重灌幂等",
    ),
    (
        "doc_id",
        "VARCHAR",
        {"max_length": 64},
        "文档 id = tb_document_parse_job.id：删除、引用回查都靠它",
    ),
    (
        "chunk_index",
        "INT32",
        {},
        "块在文档内的序号：排序、邻块扩展、去重",
    ),
    (
        "page",
        "INT32",
        {"nullable": True},
        "页码：PDF 有；docx 转 Markdown 后无页概念，写 NULL",
    ),
    (
        "content",
        "VARCHAR",
        {"max_length": MAX_CONTENT_BYTES, "enable_analyzer": True},
        "块正文：同时是 BM25 输入，analyzer 由 --analyzer 决定",
    ),
    (
        "embedding",
        "FLOAT_VECTOR",
        {"dim": DEFAULT_DIM},
        "稠密语义向量：维度由 --dim 决定，换模型必须新建集合",
    ),
    (
        "sparse",
        "SPARSE_FLOAT_VECTOR",
        {},
        "BM25 稀疏向量：由内置函数写入，客户端不要赋值",
    ),
)

# (字段名, 索引类型, 额外参数)
INDEX_SPECS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("embedding", "HNSW", {"params": {"M": 16, "efConstruction": 200}}),
    (
        "sparse",
        "SPARSE_INVERTED_INDEX",
        {
            "metric_type": "BM25",
            "params": {
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        },
    ),
    ("doc_id", "INVERTED", {}),
)


def _require_pymilvus() -> Any:
    try:
        import pymilvus
    except ImportError:  # pragma: no cover - 取决于环境
        sys.exit("缺少 pymilvus：请先 `uv sync --extra milvus` 或 `pip install 'pymilvus>=2.5'`")
    if not hasattr(pymilvus, "Function"):
        sys.exit("BM25 全文检索需要 pymilvus>=2.5，请升级后重试")
    return pymilvus


def build_schema(pymilvus: Any, *, dim: int, analyzer: str) -> Any:
    """按 FIELD_SPECS 组装 CollectionSchema。"""
    DataType = pymilvus.DataType
    schema = pymilvus.MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)

    for name, type_name, params, _note in FIELD_SPECS:
        kwargs = dict(params)
        if type_name == "FLOAT_VECTOR":
            kwargs["dim"] = dim
        if name == "content":
            kwargs["analyzer_params"] = dict(ANALYZER_PARAMS[analyzer])
        schema.add_field(name, getattr(DataType, type_name), **kwargs)

    schema.add_function(
        pymilvus.Function(
            name="content_bm25",
            input_field_names=["content"],
            output_field_names=["sparse"],
            function_type=pymilvus.FunctionType.BM25,
        )
    )
    return schema


def build_index_params(pymilvus: Any, *, metric_type: str) -> Any:
    """按 INDEX_SPECS 组装索引参数（稠密向量的 metric 由 --metric 决定）。"""
    index_params = pymilvus.MilvusClient.prepare_index_params()
    for field_name, index_type, extra in INDEX_SPECS:
        kwargs = dict(extra)
        if field_name == "embedding":
            kwargs["metric_type"] = metric_type
        index_params.add_index(field_name=field_name, index_type=index_type, **kwargs)
    return index_params


def spec_dict(
    *,
    collection: str,
    dim: int,
    metric_type: str,
    analyzer: str,
    database: str = "",
    consistency_level: str = DEFAULT_CONSISTENCY,
) -> dict[str, Any]:
    """给 --print-spec 用的纯 JSON 描述（不依赖 pymilvus）。"""
    return {
        "database": database or "default",
        "collection": collection,
        "dim": dim,
        "metric_type": metric_type,
        "analyzer": analyzer,
        "analyzer_params": ANALYZER_PARAMS[analyzer],
        "consistency_level": consistency_level,
        "enable_dynamic_field": False,
        "doc_metadata_source": "MySQL tb_document_parse_job（按 doc_id 回查）",
        "fields": [
            {"name": name, "type": type_name, "params": params, "note": note}
            for name, type_name, params, note in FIELD_SPECS
        ],
        "functions": [
            {
                "name": "content_bm25",
                "type": "BM25",
                "input": ["content"],
                "output": ["sparse"],
            }
        ],
        "indexes": [
            {"field": name, "index_type": index_type, "params": extra}
            for name, index_type, extra in INDEX_SPECS
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="创建 Milvus 检索块表 rag_chunk（7 字段 + 3 索引 + BM25 函数）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    parser.add_argument("--token", default=os.getenv("MILVUS_TOKEN", ""))
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help=f"目标数据库，默认 {DEFAULT_DATABASE}；写 default 回到默认库",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="稠密向量维度")
    parser.add_argument("--metric", default=DEFAULT_METRIC, choices=("COSINE", "IP", "L2"))
    parser.add_argument(
        "--analyzer",
        default=DEFAULT_ANALYZER,
        choices=tuple(ANALYZER_PARAMS),
        help="正文分词器（建表后不可改）",
    )
    parser.add_argument(
        "--consistency",
        default=DEFAULT_CONSISTENCY,
        choices=("Strong", "Session", "Bounded", "Eventually"),
    )
    parser.add_argument("--recreate", action="store_true", help="集合已存在时先删除（危险）")
    parser.add_argument("--print-spec", action="store_true", help="只打印字段/索引 JSON，不连库")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dim <= 0:
        sys.exit("--dim 必须为正整数")

    if args.print_spec:
        payload = spec_dict(
            collection=args.collection,
            dim=args.dim,
            metric_type=args.metric,
            analyzer=args.analyzer,
            database=args.database,
            consistency_level=args.consistency,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    pymilvus = _require_pymilvus()
    client_kwargs: dict[str, Any] = {"uri": args.uri}
    if args.token:
        client_kwargs["token"] = args.token
    client = pymilvus.MilvusClient(**client_kwargs)

    if args.database:
        if args.database not in client.list_databases():
            print(
                f"数据库 {args.database} 不存在；先建库：\n"
                f"  uv run python docs/milvus/create_database.py --name {args.database}",
                file=sys.stderr,
            )
            return 1
        client.use_database(args.database)
        print(f"目标数据库：{args.database}")

    if client.has_collection(args.collection):
        if not args.recreate:
            print(
                f"集合 {args.collection} 已存在；重建请加 --recreate（会删除现有数据）",
                file=sys.stderr,
            )
            return 1
        client.drop_collection(args.collection)
        print(f"已删除旧集合 {args.collection}")

    client.create_collection(
        collection_name=args.collection,
        schema=build_schema(pymilvus, dim=args.dim, analyzer=args.analyzer),
        index_params=build_index_params(pymilvus, metric_type=args.metric),
        consistency_level=args.consistency,
    )

    print(
        f"已创建 {args.database or 'default'}.{args.collection}：{len(FIELD_SPECS)} 个字段，"
        f"{len(INDEX_SPECS)} 个索引，dim={args.dim}，{args.metric}，analyzer={args.analyzer}"
    )
    print("索引：", ", ".join(client.list_indexes(args.collection)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
