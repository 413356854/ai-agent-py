"""LangGraph 检查点持久化：单文件 SQLite（HITL / 断点续跑的前提）。

``interrupt()`` 暂停后，图的完整状态（各 channel 的取值、待执行的节点、未消费的
写入）必须落盘，进程重启后还能用 ``Command(resume=...)`` 接着跑。LangGraph 自带的
``InMemorySaver`` 只活在进程里，重启即丢；官方 ``SqliteSaver`` 需要额外依赖
（``langgraph-checkpoint-sqlite``），这里用标准库 ``sqlite3`` 实现同一套
:class:`BaseCheckpointSaver` 接口，因此：

* **零新依赖**：只用标准库，离线测试与部署都不需要装东西；
* **单文件**：默认 ``data/agent_runs.sqlite3``，与 ``data/parsed`` 一样是本地运行态资产；
* **多进程安全**：每次操作短连接 + WAL + ``busy_timeout``，uvicorn 多 worker 也能共享。

表结构照搬 ``InMemorySaver`` 的三层存储，只是换成了行：

* ``checkpoints`` — 一次检查点（``checkpoint`` 与 ``metadata`` 各自序列化）；
* ``checkpoint_blobs`` — 每个 channel 每个版本的值（大对象按 channel 拆开存）；
* ``checkpoint_writes`` — 一个 super-step 内的待消费写入（``task_id`` + ``idx`` 唯一）。

序列化用 LangGraph 的 ``JsonPlusSerializer``（``self.serde``），与官方 saver 一致。
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

#: 三张表：检查点 / channel 取值 / 待消费写入。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT NOT NULL,
    checkpoint BLOB NOT NULL,
    metadata_type TEXT NOT NULL,
    metadata BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    value BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT NOT NULL,
    value BLOB NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
"""

#: 单个 channel 的 blob 键（线程、命名空间、通道、版本）。
_BlobKey = tuple[str, str, str, str]

#: SQLite 忙等超时（毫秒）：别的 worker 正在写时排队，而不是直接报 locked。
_BUSY_TIMEOUT_MS = 5000


class SqliteCheckpointer(BaseCheckpointSaver[str]):
    """把 LangGraph 检查点写进一个 SQLite 文件（同步接口，线程安全）。

    与官方 ``InMemorySaver`` 的语义保持一致：``put`` 覆盖同 id 的检查点，
    ``put_writes`` 对普通写入（``idx >= 0``）只写一次、对 ``__interrupt__`` 等
    特殊写入（负 ``idx``）允许覆盖；``get_tuple`` 返回最近一次检查点（或指定 id）。
    """

    def __init__(self, path: str | Path, *, serde: Any | None = None) -> None:
        super().__init__(serde=serde)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            # WAL 让「读」不被「写」挡住；设置会写进库文件，后续连接自动沿用。
            connection.execute("PRAGMA journal_mode=WAL").fetchone()
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(_SCHEMA)
            connection.commit()

    # ---- 连接 -----------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """每次操作一个短连接：线程池里哪个线程调用都不会踩 ``check_same_thread``。"""
        connection = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        try:
            yield connection
        finally:
            connection.close()

    # ---- 读取 -----------------------------------------------------------
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """读一个检查点：带 ``checkpoint_id`` 时读指定版本，否则读该线程最新版本。"""
        thread_id, checkpoint_ns = _scope(config)
        checkpoint_id = get_checkpoint_id(config)
        with self._connect() as connection:
            if checkpoint_id:
                row = connection.execute(
                    "SELECT * FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? "
                    "AND checkpoint_id = ?",
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? "
                    "ORDER BY checkpoint_id DESC LIMIT 1",
                    (thread_id, checkpoint_ns),
                ).fetchone()
            if row is None:
                return None
            return self._tuple(connection, row)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """按线程 / 命名空间倒序列出检查点（时间回溯与 ``get_state_history`` 用）。"""
        clauses: list[str] = []
        params: list[Any] = []
        if config is not None:
            clauses.append("thread_id = ?")
            params.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                clauses.append("checkpoint_ns = ?")
                params.append(checkpoint_ns)
            checkpoint_id = get_checkpoint_id(config)
            if checkpoint_id:
                clauses.append("checkpoint_id = ?")
                params.append(checkpoint_id)
        before_id = get_checkpoint_id(before) if before else None
        if before_id:
            clauses.append("checkpoint_id < ?")
            params.append(before_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM checkpoints"
                f"{where} ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC",
                params,
            ).fetchall()
            remaining = limit
            for row in rows:
                metadata = self.serde.loads_typed((row["metadata_type"], row["metadata"]))
                if filter and not all(
                    metadata.get(key) == value for key, value in filter.items()
                ):
                    continue
                if remaining is not None:
                    if remaining <= 0:
                        return
                    remaining -= 1
                yield self._tuple(connection, row)

    def _tuple(self, connection: sqlite3.Connection, row: sqlite3.Row) -> CheckpointTuple:
        """行 → `CheckpointTuple`：把 channel 取值与待消费写入一起装配好。"""
        thread_id = str(row["thread_id"])
        checkpoint_ns = str(row["checkpoint_ns"])
        checkpoint_id = str(row["checkpoint_id"])
        parent_id = row["parent_checkpoint_id"]
        stored: Checkpoint = self.serde.loads_typed((row["type"], row["checkpoint"]))
        checkpoint: Checkpoint = {
            **stored,
            "channel_values": self._load_blobs(
                connection, thread_id, checkpoint_ns, stored.get("channel_versions") or {}
            ),
        }
        metadata = self.serde.loads_typed((row["metadata_type"], row["metadata"]))
        pending = [
            (task_id, channel, self.serde.loads_typed((type_, value)))
            for task_id, channel, type_, value in connection.execute(
                "SELECT task_id, channel, type, value FROM checkpoint_writes "
                "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
                "ORDER BY task_path, task_id, idx",
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchall()
        ]
        current: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": str(parent_id),
                }
            }
            if parent_id
            else None
        )
        return CheckpointTuple(
            config=current,
            checkpoint=checkpoint,
            metadata=metadata,
            pending_writes=pending or None,
            parent_config=parent_config,
        )

    def _load_blobs(
        self,
        connection: sqlite3.Connection,
        thread_id: str,
        checkpoint_ns: str,
        versions: dict[str, Any],
    ) -> dict[str, Any]:
        """按 ``channel_versions`` 取回每个 channel 当时的取值。"""
        values: dict[str, Any] = {}
        for channel, version in versions.items():
            row = connection.execute(
                "SELECT type, value FROM checkpoint_blobs "
                "WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?",
                (thread_id, checkpoint_ns, channel, _version_key(version)),
            ).fetchone()
            if row is None or row[0] == "empty":
                continue
            values[channel] = self.serde.loads_typed((row[0], row[1]))
        return values

    # ---- 写入 -----------------------------------------------------------
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """写一次检查点，并把 ``new_versions`` 里的 channel 取值落成 blob。"""
        thread_id, checkpoint_ns = _scope(config)
        copy = dict(checkpoint)
        values: dict[str, Any] = copy.pop("channel_values", {})  # type: ignore[assignment]
        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(copy)
        metadata_type, metadata_blob = self.serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )
        with self._lock, self._connect() as connection:
            for channel, version in new_versions.items():
                if channel in values:
                    type_, blob = self.serde.dumps_typed(values[channel])
                else:
                    type_, blob = "empty", b""
                connection.execute(
                    "INSERT OR REPLACE INTO checkpoint_blobs "
                    "(thread_id, checkpoint_ns, channel, version, type, value) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (thread_id, checkpoint_ns, channel, _version_key(version), type_, blob),
                )
            connection.execute(
                "INSERT OR REPLACE INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
                " type, checkpoint, metadata_type, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    checkpoint_ns,
                    str(checkpoint["id"]),
                    config["configurable"].get("checkpoint_id"),
                    checkpoint_type,
                    checkpoint_blob,
                    metadata_type,
                    metadata_blob,
                ),
            )
            connection.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": str(checkpoint["id"]),
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """写一个 super-step 的待消费写入；``__interrupt__`` 等特殊写入可被覆盖。"""
        thread_id, checkpoint_ns = _scope(config)
        checkpoint_id = config["configurable"]["checkpoint_id"]
        with self._lock, self._connect() as connection:
            for position, (channel, value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel, position)
                type_, blob = self.serde.dumps_typed(value)
                if idx >= 0:
                    # 普通写入只保留第一次：重放同一个 super-step 不覆盖已有结果。
                    connection.execute(
                        "INSERT OR IGNORE INTO checkpoint_writes "
                        "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, "
                        " channel, type, value, task_path) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            idx,
                            channel,
                            type_,
                            blob,
                            task_path,
                        ),
                    )
                else:
                    connection.execute(
                        "INSERT OR REPLACE INTO checkpoint_writes "
                        "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, "
                        " channel, type, value, task_path) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            idx,
                            channel,
                            type_,
                            blob,
                            task_path,
                        ),
                    )
            connection.commit()

    def delete_thread(self, thread_id: str) -> None:
        """删除一个线程的全部检查点、blob 与待消费写入（取消运行后清理）。"""
        with self._lock, self._connect() as connection:
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                connection.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            connection.commit()

    def close(self) -> None:
        """没有长连接要关；保留这个方法方便调用方统一生命周期。"""


def _scope(config: RunnableConfig) -> tuple[str, str]:
    """从 RunnableConfig 里取 ``(thread_id, checkpoint_ns)``。"""
    configurable = config["configurable"]
    return str(configurable["thread_id"]), str(configurable.get("checkpoint_ns", ""))


def _version_key(version: Any) -> str:
    """channel 版本 → 文本主键（int / str 两种版本号都要能存）。"""
    return str(version)
