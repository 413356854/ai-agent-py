"""Typed application settings, loaded from `.env` / environment variables.

Every knob of the pipeline lives here so the rest of the code never reads
`os.environ` directly. Precedence: real env vars > `.env` file > defaults.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from ai_agent_py.db.dsn import MySQLTarget

PROJECT_ROOT = Path(os.getenv("AI_AGENT_PY_ROOT", Path(__file__).resolve().parents[2])).resolve()

MineruModelVersion = Literal["pipeline", "vlm", "MinerU-HTML"]


def _mask(secret: str | None) -> str:
    if not secret:
        return "missing"
    return f"set (***{secret[-4:]})" if len(secret) > 4 else "set"


class Settings(BaseSettings):
    """Runtime configuration for the whole application."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- application ---------------------------------------------------
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # ---- chat model: any OpenAI compatible endpoint --------------------
    # OpenAI / DeepSeek / Qwen / Moonshot / Ollama / vLLM all work here.
    llm_model: str = "gpt-4o-mini"
    # 通用 Agent 专用模型：要接收用户上传图片时必须支持视觉（如 qwen3-vl-plus）；
    # 留空跟随 LLM_MODEL。其他链路（RAG 问答等）仍使用 LLM_MODEL。
    general_agent_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int | None = None
    llm_timeout: float = 60.0
    # ---- 智能体循环预算（graph/budget.py，A-02）--------------------------
    # AGENT_MAX_TURNS 是「步数预算」：单轮最多调用模型几次；用完后走 finalize 节点
    # 基于已有资料作答，而不是抛 GraphRecursionError。recursion_limit 只是兜底
    # （= 预算步数 × 2 + 2），两者取较大值。
    agent_max_turns: int = 8
    # 单轮最多真正执行多少次工具调用；同一个调用被跳过（重复 / 超预算）不计入。
    agent_max_tool_calls: int = 24
    # 单轮 token 预算；0 = 不限制。只有 provider 返回 usage 时才能统计。
    agent_max_tokens: int = 0
    # 单轮墙钟预算（秒）；0 = 不限制。只在 agent 节点返回后检查，不会中断正在跑的工具。
    agent_max_seconds: float = 600.0

    # ---- 人工审批（HITL，runs/service.py）--------------------------------
    # 图停在审批节点后最多等多久（秒）；0 = 不限时（一直等人，适合演示）。
    agent_approval_timeout: float = 1800.0
    # 到期仍未审批时先「升级」并延长多久（秒）；0 = 到点直接自动拒绝。
    agent_approval_escalate_seconds: float = 600.0
    # 审批暂停时的 LangGraph 检查点文件（SQLite）；重启后仍可恢复。
    agent_run_checkpoint_path: Path = PROJECT_ROOT / "data" / "agent_runs.sqlite3"

    # ---- 通用 agent 工具（read / write / edit / glob / grep / shell）----
    # 文件工具只能读写这个目录内的路径，shell 的 cwd 也限制在这里。
    agent_workspace: Path = PROJECT_ROOT
    # shell 命令默认超时（秒），单次调用可用 timeout 参数覆盖。
    agent_shell_timeout: float = 120.0
    # shell 的 stdout / stderr 各自保留的最大字符数，超出按头尾截断。
    agent_shell_max_output: int = 20000
    # ---- 工具调用策略（tools/policy.py；所有工具统一生效）------------------
    # 单次工具调用的硬超时（秒）。shell 例外：它的截止时间由命令参数控制。
    agent_tool_timeout: float = 60.0
    # 可重试工具（知识库检索 / 联网搜索 / 网页抓取）遇到网络、429、5xx 时的重试次数。
    agent_tool_retries: int = 1
    # 同一轮多个工具调用是否并行；write / edit / shell 始终串行。
    agent_tool_parallel: bool = True

    # ---- 通用 agent 的会话记忆（MySQL tb_agent_chat_session / tb_agent_chat_message）----
    # 会话列表默认返回条数；历史候选池见下方 AGENT_MEMORY_CANDIDATES。
    agent_memory_sessions: int = 50

    # ---- 通用 agent 上下文压缩（L1 工具预算 / L2 历史预算 / L3 摘要）----------
    # 模型上下文窗口与输出预留：历史预算 = 窗口 - 输出预留 - 安全余量 - 系统提示 - 工具 schema；
    # AGENT_CONTEXT_MAX_TOKENS > 0 时再取两者较小值。
    agent_model_context_tokens: int = 32768
    agent_context_max_tokens: int = 0
    agent_output_reserve_tokens: int = 2048
    agent_context_safety_tokens: int = 512
    # 中文按「字符 / token」估算的密度；越小越保守（默认 2.0，英文可用 4.0）。
    agent_token_chars: float = 2.0
    # 从 MySQL 读取的候选历史条数（再按 token 预算裁剪）。
    agent_memory_candidates: int = 200
    # 最近多少条消息优先保留原文（预算极紧时仍可能被裁剪）。
    agent_history_keep_messages: int = 8
    # 单条历史消息进入提示词的最大字符数（中间截断）。
    agent_message_max_chars: int = 8000
    # 工具输出：单次回给模型的上限（保留头 + 尾，中间省略）；单轮累计上限。
    agent_tool_result_max_chars: int = 8000
    agent_tool_result_head_chars: int = 4096
    agent_tool_result_tail_chars: int = 1024
    agent_tool_turn_max_chars: int = 20000
    # 滚动摘要：未压缩历史超过 trigger 时，把更早部分合并成摘要。
    agent_summary_enabled: bool = True
    agent_summary_trigger_tokens: int = 12000
    agent_summary_keep_messages: int = 8
    agent_summary_max_chars: int = 1200
    agent_summary_max_tokens: int = 1200
    agent_summary_input_max_chars: int = 30000
    # 上下文溢出（provider 返回 context length 错误）后的强制压缩重试次数。
    agent_context_overflow_retries: int = 1

    # ---- 联网搜索（阿里云百炼 Web Search，DashScope 原生接口）------------
    # 文档：https://help.aliyun.com/zh/model-studio/web-search
    # Key 留空时回退 DASHSCOPE_API_KEY；两者都没有时工具调用返回配置提示。
    # 走原生接口是因为 OpenAI 兼容接口开了搜索也拿不到来源列表。
    web_search_api_key: SecretStr | None = None
    web_search_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    web_search_model: str = "qwen-plus"
    # 搜索量级策略；留空用服务端默认（turbo）。可选 turbo / max / agent / agent_max。
    web_search_strategy: str | None = None
    # 强制搜索：不让模型因为「觉得自己知道」而跳过联网。
    web_search_forced: bool = True
    web_search_enable_citation: bool = True
    web_search_timeout: float = 30.0

    # ---- 网页抓取（阿里云百炼 Web Extractor，OpenAI 兼容 Responses API）--
    # 文档：https://help.aliyun.com/zh/model-studio/web-extractor
    # 原生 Chat Completions 只支持流式；Responses API 非流式即可返回 output
    # 列表（web_extractor_call + 最终正文）。LLM_BASE_URL 是百炼域名时默认复用
    # 它的端点 / Key / 模型，否则退回百炼公网端点 + 联网搜索 Key。
    web_extractor_base_url: str | None = None
    web_extractor_api_key: SecretStr | None = None
    web_extractor_model: str | None = None
    web_extractor_timeout: float = 120.0
    # 回填给模型的正文上限（字符），避免整页内容塞爆上下文。
    web_extractor_max_chars: int = 20000

    # ---- embeddings ----------------------------------------------------
    # 阿里云百炼（DashScope）OpenAI 兼容模式：text-embedding-v4，默认 1024 维，
    # 可通过 EMBEDDING_DIMENSIONS 指定 1536/1024/768...，必须与 MILVUS_DIM 一致。
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: SecretStr | None = None
    dashscope_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
    embedding_dimensions: int | None = None
    embedding_batch_size: int = 64
    # `None` -> auto: only OpenAI hosts get tiktoken based chunking.
    embedding_check_ctx_length: bool | None = None

    # ---- Milvus（唯一向量库：通用业务 RAG 表，建表脚本 docs/milvus/create_rag_chunk.py）----
    # 写入在 `rag.milvus`，混合检索在 `rag.milvus_search`；建表后由 Milvus 保存向量与正文。
    milvus_uri: str = "http://localhost:19530"
    milvus_token: SecretStr | None = None
    milvus_database: str = "db_tax_rag"
    milvus_collection: str = "rag_chunk"
    # 必须与 EMBEDDING_MODEL 的输出维度一致；换模型 = 建新集合 + 重建索引。
    milvus_dim: int = 1536
    milvus_metric: Literal["COSINE", "IP", "L2"] = "COSINE"
    # 单租户部署写 default 即可；多租户时 tenant 是分区键。
    milvus_tenant: str = "default"
    milvus_kb: str = "default"
    milvus_num_partitions: int = 16
    # 中文文档用 chinese（jieba 分词）；纯英文语料用 english；混合语料可试 standard。
    milvus_analyzer: Literal["chinese", "english", "standard"] = "chinese"
    milvus_batch_size: int = 500
    milvus_candidates: int = 30
    milvus_ef: int = 64
    milvus_rrf_k: int = 60

    # ---- rerank（RRF 融合之后的精排）------------------------------------
    # 阿里云百炼 qwen3.7-text-rerank：交叉编码器，直接给 (query, document) 打 0~1 分。
    # 链路：两路召回各 MILVUS_CANDIDATES 条 → RRF 融合取 RERANK_CANDIDATES 条
    #       → 精排取 RETRIEVER_K 条。没配 Key 时自动跳过（等价于只做 RRF）。
    rerank_enabled: bool = True
    rerank_model: str = "qwen3.7-text-rerank"
    rerank_api_key: SecretStr | None = None
    # 完整端点；留空按 EMBEDDING_BASE_URL 推导（百炼 MaaS 域名），否则用公共网关。
    rerank_endpoint: str | None = None
    # 进入精排的候选条数：太小精排没有腾挪空间，太大拖慢响应且更费 token。
    rerank_candidates: int = 30
    # 0~1；只在单次请求内可比（百炼文档明确说明），留空表示不按分数丢弃。
    rerank_min_score: float | None = None
    # 任务说明，仅 qwen3 系列 rerank 支持；留空用模型默认的问答检索任务。
    rerank_instruct: str | None = None
    rerank_timeout: float = 10.0

    # ---- ingestion -----------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- retrieval -----------------------------------------------------
    # 最终交给模型的片段数（Milvus 侧先召回 MILVUS_CANDIDATES 条再 RRF/精排）。
    retriever_k: int = 4

    # ---- MinerU 精准解析 API -------------------------------------------
    # 文档地址：https://mineru.net/apiManage/docs
    mineru_api_key: SecretStr | None = None
    mineru_base_url: str = "https://mineru.net/api/v4"
    mineru_model_version: MineruModelVersion = "vlm"
    mineru_language: str = "ch"
    mineru_enable_formula: bool = True
    mineru_enable_table: bool = True
    mineru_is_ocr: bool = False
    # 单个任务最长等待时间（秒）与轮询间隔（秒）
    mineru_timeout: float = 600.0
    mineru_poll_interval: float = 3.0
    # 本地同时进行的解析任务上限
    mineru_max_concurrent: int = 2
    mineru_max_file_mb: int = 200
    mineru_output_dir: Path = PROJECT_ROOT / "data" / "parsed"

    # ---- RAG 切片补删队列（删除对称的兜底）-------------------------------
    # 删除文档时若 Milvus 不可用，待删的 doc_id 落盘到这里，后台重试；目录里是
    # 一个 doc_id 一个 JSON 标记，成功一条删一条（见 `rag.deletions`）。
    rag_delete_pending_dir: Path = PROJECT_ROOT / "data" / "rag_delete_pending"

    # ---- 阿里云 OSS（私域图片：管理接口走 access，图片读取走 OSS 直链）----
    # Bucket 是私有的：应用用 AccessKey 在服务端做 V1 签名，前端永远拿不到
    # AccessKey Secret。上传 / 列表 / 删除 / 换取直链要带 access 口令
    # （`X-OSS-Access` 请求头或 `?access=`）；图片统一用 OSS 预签名直链读取。
    oss_bucket: str | None = None
    # 区域 Endpoint，如 https://oss-cn-shenzhen.aliyuncs.com（不带 Bucket）。
    oss_endpoint: str | None = None
    oss_region: str = "cn-shenzhen"
    oss_access_key_id: SecretStr | None = None
    oss_access_key_secret: SecretStr | None = None
    # 对象键前缀（对应控制台里的目录），上传写 `<prefix>/<年>/<月>/<日>/<uuid>.<后缀>`。
    oss_prefix: str = "image3"
    # 访问口令：上传 / 列表 / 删除 / 换取直链必须携带；留空时管理接口返回 503。
    oss_access_token: SecretStr | None = None
    # OSS 直链（预览 / 附件下载）的有效期（秒）。
    oss_url_expires: int = 86400
    # 单张图片大小上限与允许的后缀（逗号分隔）。
    oss_max_file_mb: int = 10
    oss_allowed_suffixes: str = ".jpg,.jpeg,.png,.gif,.webp,.bmp"
    oss_timeout: float = 60.0

    # ---- MySQL（本地已有库；只做连接与健康检查）------------------------
    # 既接受 Java 项目的 JDBC 形式（jdbc:mysql://...），也接受 SQLAlchemy 的
    # mysql+pymysql://... 形式；连接参数由 `db.dsn` 统一翻译。
    mysql_url: str | None = None
    mysql_username: str | None = None
    mysql_password: SecretStr | None = None
    mysql_pool_size: int = 5
    mysql_max_overflow: int = 10
    mysql_pool_recycle: int = 1800
    mysql_connect_timeout: int = 5

    # ---- derived helpers -----------------------------------------------
    @property
    def mysql_target(self) -> MySQLTarget | None:
        """解析后的 MySQL 连接目标；未配置时返回 `None`。"""
        if not self.mysql_url:
            return None
        from ai_agent_py.db.dsn import parse_mysql_url

        return parse_mysql_url(
            self.mysql_url,
            username=self.mysql_username,
            password=self.mysql_password.get_secret_value() if self.mysql_password else None,
            connect_timeout=self.mysql_connect_timeout,
        )

    @property
    def mysql_dsn(self) -> str | None:
        """SQLAlchemy 连接串（含密码，只应交给引擎，不要写日志）。"""
        target = self.mysql_target
        return target.dsn if target else None

    @property
    def mysql_safe_dsn(self) -> str | None:
        """可展示的连接串（密码显示为 `***`）。"""
        target = self.mysql_target
        return target.safe_dsn if target else None

    # ---- OSS 派生属性 ----------------------------------------------------
    @property
    def resolved_oss_access_key_id(self) -> str | None:
        return self.oss_access_key_id.get_secret_value() if self.oss_access_key_id else None

    @property
    def resolved_oss_access_key_secret(self) -> str | None:
        if self.oss_access_key_secret is None:
            return None
        return self.oss_access_key_secret.get_secret_value()

    @property
    def resolved_oss_access_token(self) -> str | None:
        return self.oss_access_token.get_secret_value() if self.oss_access_token else None

    @property
    def oss_configured(self) -> bool:
        """Bucket / Endpoint / AccessKey 都齐了才算可读写 OSS。"""
        return bool(
            self.oss_bucket
            and self.oss_endpoint
            and self.resolved_oss_access_key_id
            and self.resolved_oss_access_key_secret
        )

    @property
    def oss_access_configured(self) -> bool:
        return bool(self.resolved_oss_access_token)

    @property
    def oss_endpoint_url(self) -> str | None:
        return self.oss_endpoint.rstrip("/") if self.oss_endpoint else None

    @property
    def oss_host(self) -> str | None:
        """虚拟主机风格访问域名：``<bucket>.<endpoint-host>``。"""
        if not self.oss_endpoint:
            return None
        host = urlsplit(self.oss_endpoint).hostname
        if not host:
            return None
        if not self.oss_bucket or host.startswith(f"{self.oss_bucket}."):
            return host
        return f"{self.oss_bucket}.{host}"

    @property
    def oss_base_url(self) -> str | None:
        host = self.oss_host
        return f"https://{host}" if host else None

    @property
    def oss_suffixes(self) -> tuple[str, ...]:
        return tuple(item.strip() for item in self.oss_allowed_suffixes.split(",") if item.strip())

    @property
    def resolved_llm_api_key(self) -> str | None:
        """对话模型 Key：``LLM_API_KEY`` → ``DASHSCOPE_API_KEY`` → ``OPENAI_API_KEY``。

        阿里云百炼同时提供对话与向量接口，很多环境只导出了 ``DASHSCOPE_API_KEY``；
        把它作为 ``LLM_API_KEY`` 的第二个来源（环境变量或 ``.env`` 均可），
        ``serve`` / ``ai-agent-py chat`` 就不必重复导出同一个 Key。
        ``OPENAI_API_KEY`` 仍作为最后的兜底，保持原有行为。
        """
        if self.llm_api_key is not None:
            return self.llm_api_key.get_secret_value()
        if self.dashscope_api_key is not None:
            return self.dashscope_api_key.get_secret_value()
        return os.getenv("OPENAI_API_KEY") or None

    @property
    def resolved_embedding_api_key(self) -> str | None:
        """Embedding key: dedicated key, else DashScope, else OpenAI, else chat key."""
        if self.embedding_api_key is not None:
            return self.embedding_api_key.get_secret_value()
        if self.dashscope_api_key is not None:
            return self.dashscope_api_key.get_secret_value()
        return os.getenv("OPENAI_API_KEY") or self.resolved_llm_api_key

    @property
    def is_dashscope_embedding(self) -> bool:
        """``EMBEDDING_BASE_URL`` 是否指向百炼（含业务空间 MaaS 域名）。"""
        host = urlsplit(self.embedding_base_url or "").hostname or ""
        return host.endswith("aliyuncs.com") and ("dashscope" in host or ".maas." in host)

    @property
    def resolved_rerank_api_key(self) -> str | None:
        """Rerank key：专用 Key → 百炼 Key → 同业务空间的 embedding Key。

        精排走的是百炼 ``/api/v1/services/rerank/...``，所以：
        * 只有 Key 属于百炼时才能复用；``EMBEDDING_API_KEY`` 仅在
          ``EMBEDDING_BASE_URL`` 也指向百炼域名时才算数，避免把 OpenAI /
          硅基流动的 Key 发到阿里云去（那只会换来一个 401 和一个无用的告警）；
        * 永远不回退 ``LLM_API_KEY``：对话模型可能接的是 DeepSeek。
        """
        if self.rerank_api_key is not None:
            return self.rerank_api_key.get_secret_value()
        if self.dashscope_api_key is not None:
            return self.dashscope_api_key.get_secret_value()
        if self.embedding_api_key is not None and self.is_dashscope_embedding:
            return self.embedding_api_key.get_secret_value()
        return None

    @property
    def resolved_mineru_api_key(self) -> str | None:
        """MinerU token, falling back to the original `MINERU_TOKEN` name."""
        if self.mineru_api_key is not None:
            return self.mineru_api_key.get_secret_value()
        return os.getenv("MINERU_TOKEN") or None

    @property
    def resolved_milvus_token(self) -> str | None:
        """Milvus / Zilliz Cloud 的 API Key 或 `user:password` 形式 token。"""
        return self.milvus_token.get_secret_value() if self.milvus_token else None

    @property
    def resolved_web_search_api_key(self) -> str | None:
        """联网搜索 Key：``WEB_SEARCH_API_KEY`` → ``DASHSCOPE_API_KEY``。

        联网搜索走 DashScope 原生接口，不跟着 ``LLM_API_KEY`` 走：对话模型可能
        接的是 DeepSeek / 自建 vLLM，把那个 Key 发到阿里云只会得到 401。
        """
        if self.web_search_api_key is not None:
            return self.web_search_api_key.get_secret_value()
        if self.dashscope_api_key is not None:
            return self.dashscope_api_key.get_secret_value()
        return None

    @property
    def resolved_web_extractor_base_url(self) -> str:
        """网页抓取端点：显式覆盖 → 百炼 MaaS 端点 → 百炼公网。

        对话模型不是百炼（比如 DeepSeek / 自建 vLLM）时不能复用它的 Base URL：
        ``/responses`` 与 web_extractor 只有百炼兼容端点才有，所以退回公网端点，
        Key 也随之回退到 ``WEB_SEARCH_API_KEY`` / ``DASHSCOPE_API_KEY``。
        """
        if self.web_extractor_base_url:
            return self.web_extractor_base_url
        host = urlsplit(self.llm_base_url or "").hostname or ""
        if self.llm_base_url and (host.endswith("aliyuncs.com") or host.endswith("qianwenai.com")):
            return self.llm_base_url
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @property
    def resolved_web_extractor_api_key(self) -> str | None:
        """抓取 Key：显式覆盖 → 百炼对话模型 Key → 联网搜索 Key。"""
        if self.web_extractor_api_key is not None:
            return self.web_extractor_api_key.get_secret_value()
        if self.llm_base_url and self.resolved_web_extractor_base_url == self.llm_base_url:
            return self.resolved_llm_api_key
        return self.resolved_web_search_api_key

    @property
    def resolved_web_extractor_model(self) -> str:
        """抓取模型：显式覆盖 → 对话模型（必须支持 web_extractor）。"""
        return self.web_extractor_model or self.llm_model

    def describe(self) -> dict[str, Any]:
        """Human readable snapshot with secrets masked (used by `info`)."""
        target = self.mysql_target
        mysql = (
            f"{target.username}@{target.host}:{target.port}/{target.database}"
            if target
            else "未配置"
        )
        return {
            "env": self.app_env,
            "llm": f"{self.llm_model} @ {self.llm_base_url or 'api.openai.com'}",
            "general_agent_model": self.general_agent_model or self.llm_model,
            "llm_api_key": _mask(self.resolved_llm_api_key),
            "embeddings": f"{self.embedding_model} @ {self.embedding_base_url or 'api.openai.com'}",
            "embedding_api_key": _mask(self.resolved_embedding_api_key),
            "embedding_dimensions": self.embedding_dimensions or "auto",
            "milvus": (
                f"{self.milvus_uri} / {self.milvus_database}.{self.milvus_collection} "
                f"(dim={self.milvus_dim}, {self.milvus_metric}, {self.milvus_analyzer})"
            ),
            "milvus_token": "set" if self.milvus_token else "missing",
            "data_dir": str(self.data_dir),
            "agent_workspace": str(self.agent_workspace),
            "agent_shell": (
                f"timeout={self.agent_shell_timeout:g}s, max_output={self.agent_shell_max_output}"
            ),
            "agent_tools": (
                f"timeout={self.agent_tool_timeout:g}s, retries={self.agent_tool_retries}, "
                f"parallel={'on' if self.agent_tool_parallel else 'off'}"
            ),
            "agent_budget": (
                f"steps<={self.agent_max_turns}, tool_calls<={self.agent_max_tool_calls}, "
                f"tokens={self.agent_max_tokens or 'off'}, "
                f"seconds={self.agent_max_seconds or 'off'}"
            ),
            "agent_memory": (
                f"mysql, candidates={self.agent_memory_candidates}, "
                f"sessions={self.agent_memory_sessions}"
            ),
            "agent_approval": (
                f"timeout={self.agent_approval_timeout:g}s, "
                f"escalate={self.agent_approval_escalate_seconds:g}s, "
                f"checkpoint={self.agent_run_checkpoint_path}"
            ),
            "agent_context": (
                f"model={self.agent_model_context_tokens}, "
                f"budget={self.agent_context_max_tokens or 'auto'}, "
                f"reserve={self.agent_output_reserve_tokens}, "
                f"chars/token={self.agent_token_chars:g}, "
                f"tool={self.agent_tool_result_max_chars}/{self.agent_tool_turn_max_chars}, "
                f"summary={'on' if self.agent_summary_enabled else 'off'} "
                f"(trigger={self.agent_summary_trigger_tokens})"
            ),
            "web_search": (
                f"{self.web_search_model} @ {self.web_search_base_url} "
                f"({'on' if self.resolved_web_search_api_key else 'off'})"
            ),
            "web_search_api_key": _mask(self.resolved_web_search_api_key),
            "web_extractor": (
                f"{self.resolved_web_extractor_model} @ "
                f"{self.resolved_web_extractor_base_url} "
                f"({'on' if self.resolved_web_extractor_api_key else 'off'})"
            ),
            "chunking": f"size={self.chunk_size}, overlap={self.chunk_overlap}",
            "retrieval": (
                f"hybrid(dense+bm25) + RRF, k={self.retriever_k}, "
                f"candidates={self.milvus_candidates}"
            ),
            "rerank": (
                f"{self.rerank_model}, candidates={self.rerank_candidates}, "
                f"{'on' if self.rerank_enabled and self.resolved_rerank_api_key else 'off'}"
            ),
            "rerank_api_key": _mask(self.resolved_rerank_api_key),
            "mineru": f"{self.mineru_model_version} @ {self.mineru_base_url}",
            "mineru_api_key": _mask(self.resolved_mineru_api_key),
            "mineru_output_dir": str(self.mineru_output_dir),
            "rag_delete_pending": str(self.rag_delete_pending_dir),
            "mysql": mysql,
            # 数据库口令不做“后四位”提示：短口令会被直接猜出来。
            "mysql_password": "set" if self.mysql_password else "missing",
            "oss": (
                f"{self.oss_bucket or '未配置'} @ {self.oss_host or '未配置'} / "
                f"{self.oss_prefix} ({'on' if self.oss_configured else 'off'})"
            ),
            "oss_access_key_id": _mask(self.resolved_oss_access_key_id),
            "oss_access_token": "set" if self.oss_access_configured else "missing",
        }

    # ---- normalisation -------------------------------------------------
    @field_validator(
        "llm_base_url",
        "embedding_base_url",
        "llm_api_key",
        "general_agent_model",
        "embedding_api_key",
        "dashscope_api_key",
        "mineru_api_key",
        "mysql_url",
        "mysql_username",
        "mysql_password",
        "milvus_token",
        "rerank_api_key",
        "rerank_endpoint",
        "rerank_instruct",
        "web_search_api_key",
        "web_search_strategy",
        "web_extractor_base_url",
        "web_extractor_api_key",
        "web_extractor_model",
        "oss_bucket",
        "oss_endpoint",
        "oss_access_key_id",
        "oss_access_key_secret",
        "oss_access_token",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "data_dir",
        "mineru_output_dir",
        "rag_delete_pending_dir",
        "agent_workspace",
        "agent_run_checkpoint_path",
        mode="after",
    )
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    @field_validator("mysql_pool_size", "mysql_connect_timeout", mode="after")
    @classmethod
    def _positive_mysql_number(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("MySQL 连接池大小与连接超时必须是正整数")
        return value

    @field_validator("mysql_max_overflow", "mysql_pool_recycle", mode="after")
    @classmethod
    def _non_negative_mysql_number(cls, value: int) -> int:
        if value < -1:
            raise ValueError("MySQL 溢出连接数与回收秒数不能小于 -1")
        return value

    @field_validator("mineru_base_url", mode="after")
    @classmethod
    def _strip_base_url(cls, value: str | None) -> str:
        return (value or "https://mineru.net/api/v4").rstrip("/")

    @field_validator("embedding_base_url", mode="after")
    @classmethod
    def _strip_embedding_base_url(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value else None

    @field_validator("oss_endpoint", mode="after")
    @classmethod
    def _normalize_oss_endpoint(cls, value: str | None) -> str | None:
        """接受 ``oss-cn-shenzhen.aliyuncs.com`` 或带协议 / Bucket 的写法，统一成区域 Endpoint。"""
        if not value:
            return None
        raw = value.strip()
        if "://" not in raw:
            raw = f"https://{raw}"
        host = urlsplit(raw).hostname
        if not host:
            raise ValueError("OSS_ENDPOINT 不是合法的域名或 URL")
        return f"https://{host}"

    @field_validator("oss_bucket", mode="after")
    @classmethod
    def _normalize_oss_bucket(cls, value: str | None) -> str | None:
        if not value:
            return None
        bucket = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket):
            raise ValueError("OSS_BUCKET 只能包含小写字母、数字与中划线，长度 3~63")
        return bucket

    @field_validator("oss_prefix", mode="after")
    @classmethod
    def _normalize_oss_prefix(cls, value: str) -> str:
        prefix = (value or "").strip().strip("/")
        parts = [part for part in prefix.split("/") if part]
        if not parts:
            raise ValueError("OSS_PREFIX 不能为空（私域 OSS 只允许访问该目录下的对象）")
        if any(part in {".", ".."} for part in parts):
            raise ValueError("OSS_PREFIX 不能包含 . / .. 路径片段")
        return "/".join(parts)

    @field_validator("oss_allowed_suffixes", mode="after")
    @classmethod
    def _normalize_oss_suffixes(cls, value: str) -> str:
        suffixes: list[str] = []
        for raw in (value or "").split(","):
            suffix = raw.strip().lower()
            if not suffix:
                continue
            if not suffix.startswith("."):
                suffix = f".{suffix}"
            if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
                raise ValueError(f"OSS_ALLOWED_SUFFIXES 里的后缀不合法: {raw}")
            if suffix not in suffixes:
                suffixes.append(suffix)
        if not suffixes:
            raise ValueError("OSS_ALLOWED_SUFFIXES 至少要配置一个图片后缀")
        return ",".join(suffixes)

    @field_validator("oss_url_expires", mode="after")
    @classmethod
    def _positive_oss_url_expires(cls, value: int) -> int:
        # 60 秒 ~ 30 天：太短复制出去就失效，太长等于永久公开。
        if not 60 <= value <= 30 * 24 * 3600:
            raise ValueError("OSS_URL_EXPIRES 必须在 60 ~ 2592000 秒之间")
        return value

    @field_validator("oss_max_file_mb", mode="after")
    @classmethod
    def _positive_oss_max_file(cls, value: int) -> int:
        if not 1 <= value <= 100:
            raise ValueError("OSS_MAX_FILE_MB 必须在 1 ~ 100 之间")
        return value

    @field_validator("oss_timeout", mode="after")
    @classmethod
    def _positive_oss_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("OSS_TIMEOUT 必须是正数")
        return value

    @field_validator("web_search_base_url", mode="after")
    @classmethod
    def _strip_web_search_base_url(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("WEB_SEARCH_BASE_URL 不能为空")
        return value.rstrip("/")

    @field_validator("web_extractor_base_url", mode="after")
    @classmethod
    def _strip_web_extractor_base_url(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value else None

    @field_validator("embedding_dimensions", mode="after")
    @classmethod
    def _positive_dimensions(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS 必须是正整数")
        return value

    @field_validator(
        "milvus_dim",
        "milvus_num_partitions",
        "milvus_batch_size",
        "milvus_candidates",
        "milvus_ef",
        "milvus_rrf_k",
        "rerank_candidates",
        mode="after",
    )
    @classmethod
    def _positive_milvus_number(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Milvus 维度、分区数、批大小、召回数、ef 与精排候选数必须是正整数")
        return value

    @field_validator("rerank_timeout", mode="after")
    @classmethod
    def _positive_rerank_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("RERANK_TIMEOUT 必须是正数")
        return value

    @field_validator("agent_shell_timeout", mode="after")
    @classmethod
    def _positive_shell_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("AGENT_SHELL_TIMEOUT 必须是正数")
        return value

    @field_validator("agent_shell_max_output", mode="after")
    @classmethod
    def _positive_shell_output(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("AGENT_SHELL_MAX_OUTPUT 必须是正整数")
        return value

    @field_validator("agent_tool_timeout", mode="after")
    @classmethod
    def _positive_tool_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("AGENT_TOOL_TIMEOUT 必须是正数")
        return value

    @field_validator("agent_tool_retries", mode="after")
    @classmethod
    def _non_negative_tool_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("AGENT_TOOL_RETRIES 不能为负数")
        return value

    @field_validator("agent_max_turns", "agent_max_tool_calls", mode="after")
    @classmethod
    def _positive_loop_budget(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("AGENT_MAX_TURNS / AGENT_MAX_TOOL_CALLS 必须是正整数")
        return value

    @field_validator("agent_max_tokens", mode="after")
    @classmethod
    def _non_negative_max_tokens(cls, value: int) -> int:
        if value < 0:
            raise ValueError("AGENT_MAX_TOKENS 不能为负数（0 = 不限制）")
        return value

    @field_validator("agent_max_seconds", mode="after")
    @classmethod
    def _non_negative_max_seconds(cls, value: float) -> float:
        if value < 0:
            raise ValueError("AGENT_MAX_SECONDS 不能为负数（0 = 不限制）")
        return value

    @field_validator("agent_memory_sessions", mode="after")
    @classmethod
    def _positive_agent_memory(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("AGENT_MEMORY_SESSIONS 必须是正整数")
        return value

    @field_validator("agent_approval_timeout", "agent_approval_escalate_seconds", mode="after")
    @classmethod
    def _non_negative_approval_timeout(cls, value: float) -> float:
        if value < 0:
            raise ValueError("AGENT_APPROVAL_TIMEOUT / AGENT_APPROVAL_ESCALATE_SECONDS 不能为负数")
        return value

    @field_validator(
        "agent_model_context_tokens",
        "agent_output_reserve_tokens",
        "agent_context_safety_tokens",
        "agent_memory_candidates",
        "agent_history_keep_messages",
        "agent_message_max_chars",
        "agent_tool_result_max_chars",
        "agent_tool_result_head_chars",
        "agent_tool_result_tail_chars",
        "agent_tool_turn_max_chars",
        "agent_summary_trigger_tokens",
        "agent_summary_keep_messages",
        "agent_summary_max_chars",
        "agent_summary_max_tokens",
        "agent_summary_input_max_chars",
        mode="after",
    )
    @classmethod
    def _positive_agent_context(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("通用 Agent 的上下文预算配置必须是正整数")
        return value

    @field_validator("agent_context_max_tokens", "agent_context_overflow_retries", mode="after")
    @classmethod
    def _non_negative_agent_context(cls, value: int) -> int:
        if value < 0:
            raise ValueError("AGENT_CONTEXT_MAX_TOKENS / AGENT_CONTEXT_OVERFLOW_RETRIES 不能为负数")
        return value

    @field_validator("agent_token_chars", mode="after")
    @classmethod
    def _positive_token_chars(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("AGENT_TOKEN_CHARS 必须是正数")
        return value

    @model_validator(mode="after")
    def _validate_tool_result_budget(self) -> Settings:
        if (
            self.agent_tool_result_head_chars + self.agent_tool_result_tail_chars
            >= self.agent_tool_result_max_chars
        ):
            raise ValueError(
                "AGENT_TOOL_RESULT_HEAD_CHARS + AGENT_TOOL_RESULT_TAIL_CHARS "
                "必须小于 AGENT_TOOL_RESULT_MAX_CHARS"
            )
        return self

    @field_validator("web_search_timeout", mode="after")
    @classmethod
    def _positive_web_search_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("WEB_SEARCH_TIMEOUT 必须是正数")
        return value

    @field_validator("web_extractor_timeout", mode="after")
    @classmethod
    def _positive_web_extractor_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("WEB_EXTRACTOR_TIMEOUT 必须是正数")
        return value

    @field_validator("web_extractor_max_chars", mode="after")
    @classmethod
    def _positive_web_extractor_chars(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("WEB_EXTRACTOR_MAX_CHARS 必须是正整数")
        return value

    @field_validator("rerank_min_score", mode="after")
    @classmethod
    def _rerank_score_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("RERANK_MIN_SCORE 必须在 0~1 之间（百炼精排分数是 0~1）")
        return value

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @model_validator(mode="after")
    def _validate_mysql(self) -> Settings:
        # 配置了就必须能解析：宁可启动时失败，也不要等健康检查才发现 URL 写错。
        if self.mysql_url:
            _ = self.mysql_target
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process wide (cached) settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cache, e.g. after changing env vars in tests."""
    get_settings.cache_clear()
