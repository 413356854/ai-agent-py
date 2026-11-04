# =============================================================================
# db_tax_rag —— 公司税法 RAG 的 Milvus 数据库（database）
#
# Milvus 2.2.9+ 在 collection 之上引入 database 层，用来做逻辑隔离与配额管理；
# 默认库 `default` 永远存在且不能删除。本脚本创建 db_tax_rag，供
# create_rag_chunk.py 建表使用：
#
#   uv run python docs/milvus/create_database.py
#   uv run python docs/milvus/create_rag_chunk.py --database db_tax_rag --dim 1536
#
# 约定与限制（Milvus 官方文档）：
#   * 库名规则与 collection 一致：字母/数字/下划线，首字符为字母或下划线，≤255 字符；
#   * `default` 库不可删；删库前必须先删光库内所有 collection；
#   * 集合名只在库内唯一 —— 不同库可以有同名 collection（如各建一份 rag_chunk）；
#   * 库级属性：副本数、资源组、磁盘配额、集合数上限、禁读/禁写、时区；
#   * 数量默认上限 64 个库（rootCoord.maxDatabaseNum 可调），别按租户一库一建。
#
# 执行：
#   uv run python docs/milvus/create_database.py --uri http://localhost:19530
#   uv run python docs/milvus/create_database.py --max-collections 32 --replica 2
#   uv run python docs/milvus/create_database.py --drop        # 删库（要求库内为空）
# =============================================================================
"""创建 Milvus 数据库 db_tax_rag（幂等：已存在则跳过）。"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any

DEFAULT_DB = "db_tax_rag"
NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_NAME_LENGTH = 255

# 库级属性：参数名 -> Milvus 属性名
PROPERTY_KEYS = {
    "replica": "database.replica.number",
    "resource_groups": "database.resource_groups",
    "disk_quota_mb": "database.diskQuota.mb",
    "max_collections": "database.max.collections",
}


def _require_pymilvus() -> Any:
    try:
        import pymilvus
    except ImportError:  # pragma: no cover - 取决于环境
        sys.exit("缺少 pymilvus：请先 `uv sync --extra milvus` 或 `pip install 'pymilvus>=2.5'`")
    return pymilvus


def _validate_name(name: str) -> None:
    if not name:
        sys.exit("库名不能为空")
    if len(name) > MAX_NAME_LENGTH:
        sys.exit(f"库名最长 {MAX_NAME_LENGTH} 字符，当前 {len(name)}")
    if not NAME_PATTERN.match(name):
        sys.exit(f"库名 {name!r} 不合法：只能包含字母、数字、下划线，且不能以数字开头")


def _properties(args: argparse.Namespace) -> dict[str, Any]:
    """把命令行参数转成 Milvus 的库级属性；0 / 空串表示不设置。"""
    values = {
        "replica": args.replica,
        "resource_groups": args.resource_groups,
        "disk_quota_mb": args.disk_quota_mb,
        "max_collections": args.max_collections,
    }
    return {
        PROPERTY_KEYS[key]: value for key, value in values.items() if value not in (0, "", None)
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"创建 Milvus 数据库（默认 {DEFAULT_DB}）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    parser.add_argument("--token", default=os.getenv("MILVUS_TOKEN", ""))
    parser.add_argument("--name", default=DEFAULT_DB, help="库名")
    parser.add_argument("--replica", type=int, default=0, help="库副本数，0 表示不设置")
    parser.add_argument(
        "--resource-groups",
        default="",
        help="绑定的资源组，逗号分隔（如 rg_prod），空表示不设置",
    )
    parser.add_argument("--disk-quota-mb", type=int, default=0, help="磁盘配额 MB，0 表示不设置")
    parser.add_argument(
        "--max-collections",
        type=int,
        default=0,
        help="库内 collection 数量上限，0 表示不设置",
    )
    parser.add_argument("--drop", action="store_true", help="删除该库（要求库内为空）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_name(args.name)

    pymilvus = _require_pymilvus()
    client_kwargs: dict[str, Any] = {"uri": args.uri}
    if args.token:
        client_kwargs["token"] = args.token
    client = pymilvus.MilvusClient(**client_kwargs)

    exists = args.name in client.list_databases()

    if args.drop:
        if not exists:
            print(f"数据库 {args.name} 不存在，无需删除")
            return 0
        # 先自查是否为空：Milvus 删非空库会抛 RPC 错误，提前拦掉，报错更友好。
        client.use_database(args.name)
        collections = client.list_collections()
        if collections:
            print(
                f"数据库 {args.name} 里还有 {len(collections)} 个集合："
                f"{', '.join(collections)}；请先删光集合再删库。",
                file=sys.stderr,
            )
            return 1
        try:
            client.drop_database(args.name)
        except Exception as exc:
            print(f"删库失败：{exc}", file=sys.stderr)
            return 1
        print(f"已删除数据库 {args.name}")
        return 0

    if exists:
        print(f"数据库 {args.name} 已存在，跳过创建")
    else:
        properties = _properties(args)
        try:
            if properties:
                client.create_database(db_name=args.name, properties=properties)
            else:
                client.create_database(db_name=args.name)
        except Exception as exc:
            print(f"创建数据库失败：{exc}", file=sys.stderr)
            return 1
        suffix = f"，属性 {properties}" if properties else ""
        print(f"已创建数据库 {args.name}{suffix}")

    print(f"当前数据库：{', '.join(client.list_databases())}")
    print(
        f"下一步：uv run python docs/milvus/create_rag_chunk.py --database {args.name} --dim 1536"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
