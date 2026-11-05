"""MinerU 解析任务的数据模型与状态机。

任务状态流转::

    queued ──▶ uploading ──▶ parsing ──▶ downloading ──▶ done
       └──────────────┴───────────┴──────────────┴──────▶ failed

`queued`/`uploading` 是本地阶段（申请上传链接、PUT 文件），`parsing` 之后
由 MinerU 云端完成，本地只做轮询。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class JobState(StrEnum):
    """解析任务状态（与 MinerU 的 state 语义对齐）。"""

    queued = "queued"
    uploading = "uploading"
    parsing = "parsing"
    downloading = "downloading"
    done = "done"
    failed = "failed"


TERMINAL_STATES = frozenset({JobState.done, JobState.failed})

STATE_LABELS: dict[JobState, str] = {
    JobState.queued: "排队中",
    JobState.uploading: "上传中",
    JobState.parsing: "解析中",
    JobState.downloading: "下载结果",
    JobState.done: "已完成",
    JobState.failed: "失败",
}


class ParseOptions(BaseModel):
    """一次解析请求的参数（对应 MinerU `extract/task` 的请求体字段）。"""

    model_version: str = Field(
        default="vlm", description="pipeline / vlm / MinerU-HTML；direct 表示 md/json 直接切分"
    )
    language: str = Field(default="ch", description="OCR 语言，默认 ch")
    enable_formula: bool = Field(default=True, description="公式识别")
    enable_table: bool = Field(default=True, description="表格识别")
    is_ocr: bool = Field(default=False, description="扫描件 OCR（pipeline/vlm 有效）")
    page_ranges: str | None = Field(default=None, description='页码范围，如 "1-10,12"')

    def batch_file_payload(self, name: str, data_id: str) -> dict[str, Any]:
        """批量上传接口里 `files[]` 的单个元素。"""
        item: dict[str, Any] = {"name": name, "data_id": data_id, "is_ocr": self.is_ocr}
        if self.page_ranges:
            item["page_ranges"] = self.page_ranges
        return item

    def batch_payload(self, name: str, data_id: str) -> dict[str, Any]:
        """`POST /file-urls/batch` 的完整请求体。"""
        return {
            "files": [self.batch_file_payload(name, data_id)],
            "model_version": self.model_version,
            "language": self.language,
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
        }


class RagIngestInfo(BaseModel):
    """解析结果写入 Milvus ``rag_chunk`` 后的摘要（``/rag`` 接口）。"""

    chunks: int = 0
    characters: int = 0
    collection: str = ""
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BatchUpload(BaseModel):
    """`POST /file-urls/batch` 的返回：批次 id + 每个文件的上传地址。"""

    batch_id: str
    file_urls: list[str]


class BatchItem(BaseModel):
    """`GET /extract-results/batch/{batch_id}` 里的单个文件结果。"""

    file_name: str = ""
    data_id: str = ""
    state: str = "pending"
    full_zip_url: str | None = None
    err_msg: str = ""
    extracted_pages: int = 0
    total_pages: int = 0

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> BatchItem:
        progress = raw.get("extract_progress") or {}
        return cls(
            file_name=str(raw.get("file_name") or ""),
            data_id=str(raw.get("data_id") or ""),
            state=str(raw.get("state") or "pending"),
            full_zip_url=raw.get("full_zip_url") or None,
            err_msg=str(raw.get("err_msg") or ""),
            extracted_pages=int(progress.get("extracted_pages") or 0),
            total_pages=int(progress.get("total_pages") or 0),
        )


class ParseJob(BaseModel):
    """一个文档解析任务的完整记录（同时用于 API 响应与磁盘持久化）。"""

    id: str
    file_name: str
    file_size: int = 0
    suffix: str = ""
    state: JobState = JobState.queued
    options: ParseOptions = Field(default_factory=ParseOptions)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_seconds: float | None = None

    batch_id: str | None = None
    err_msg: str = ""

    extracted_pages: int = 0
    total_pages: int = 0

    # 相对任务目录（data/parsed/<job_id>/）的文件名，避免把绝对路径暴露给前端
    source_file: str | None = None
    markdown_file: str | None = None
    zip_file: str | None = None
    markdown_chars: int = 0
    asset_count: int = 0

    #: 最近一次写入 Milvus ``rag_chunk`` 的摘要；非空即「已入库」。
    rag_ingest: RagIngestInfo | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rag_ingested(self) -> bool:
        """是否已经写入 RAG 表（Milvus）；文档解析状态仍由 ``state`` 表示。"""
        return self.rag_ingest is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def progress(self) -> float:
        """0~1 的进度；页数未知时按状态估算。"""
        if self.state is JobState.done:
            return 1.0
        if self.total_pages:
            return min(0.99, self.extracted_pages / self.total_pages)
        return {JobState.parsing: 0.35, JobState.downloading: 0.85}.get(self.state, 0.1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def state_label(self) -> str:
        return STATE_LABELS.get(self.state, self.state.value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def assets_base(self) -> str:
        """Markdown 里图片等资源的基地址（由 FastAPI 静态目录提供）。"""
        return f"/parsed/{self.id}/" if self.markdown_file else ""


class MarkdownOut(BaseModel):
    """预览接口的响应：Markdown 正文 + 图片等资源的基地址。"""

    job_id: str
    file_name: str
    markdown: str
    assets_base: str
    characters: int
    assets: int


class DocumentConfigOut(BaseModel):
    """前端初始化所需的解析配置（不含密钥）。"""

    configured: bool
    base_url: str
    model_version: str
    default_options: ParseOptions
    model_versions: list[str]
    languages: list[str]
    max_file_mb: int
    accepted_suffixes: list[str]
    output_dir: str
    # ---- 切分 / RAG 入库 ----
    rag_configured: bool = False
    embedding_model: str = ""
    rag_database: str = ""
    rag_collection: str = ""
    rag_dim: int = 0
