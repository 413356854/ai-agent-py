"""RRF 之后的精排：把候选片段交给交叉编码器重新打分。

检索链路（`rag.milvus_search` 调用）：

    Milvus 两路召回 → RRF 融合取 RERANK_CANDIDATES 条 → 精排 → 取 RETRIEVER_K 条

RRF 只用了「名次」，两个召回器的原始分数被丢掉了；精排则把 ``(query, chunk)``
成对送进交叉编码器，直接得到一个 0~1 的相关性分，对中文法条这种「关键词命中了
但语义不相关」的场景提升明显。

对接阿里云百炼的 DashScope 原生接口（**不是** OpenAI 兼容模式）：

    POST {endpoint}
    {
      "model": "qwen3.7-text-rerank",
      "input": {"query": "业务招待费扣除标准", "documents": ["…", "…"]},
      "parameters": {"top_n": 4}
    }

    → {"output": {"results": [{"index": 0, "relevance_score": 0.93}, …]},
       "usage": {"total_tokens": 79}, "request_id": "…"}

三条约束来自官方文档（https://help.aliyun.com/zh/model-studio/text-rerank-api）：

* ``relevance_score`` 只在**单次请求内**可比，不能当成跨请求的绝对阈值；
* 单次请求的 ``documents`` 有上限（qwen3 系列 500 条），所以只送
  ``RERANK_CANDIDATES`` 条，不把整个库塞进去；
* 失败必须降级：精排是锦上添花，不能因为它超时 / 401 就让整条问答链挂掉——
  调用方（`MilvusChunkSearcher`）捕获 `RerankError` 后退回 RRF 顺序。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from langchain_core.documents import Document

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import RerankError

logger = logging.getLogger(__name__)

#: 公共网关（没有业务空间域名时的兜底）。
DASHSCOPE_RERANK_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)

#: 业务空间 MaaS 域名下的固定路径：https://{WorkspaceId}.{region}.maas.aliyuncs.com
RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"


class Reranker(Protocol):
    """精排接口：`rag.milvus_search` 只依赖这个方法，测试可以注入假实现。"""

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_n: int | None = None,
    ) -> list[Document]:
        """按与 ``query`` 的相关性重排 ``documents``，返回前 ``top_n`` 条。"""
        ...


class DashScopeReranker:
    """百炼 rerank：``client`` / ``api_key`` 可注入，测试不需要真实网络。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        api_key: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._api_key = api_key

    # ---- 连接与端点 -----------------------------------------------------
    @property
    def api_key(self) -> str:
        key = self._api_key or self.settings.resolved_rerank_api_key
        if not key:
            raise RerankError("缺少精排 Key：请配置 RERANK_API_KEY 或 DASHSCOPE_API_KEY")
        return key

    @property
    def endpoint(self) -> str:
        return resolve_endpoint(self.settings)

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.settings.rerank_timeout)
        return self._client

    def close(self) -> None:
        """释放连接池（进程级单例通常不需要，测试 / CLI 收尾时用）。"""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---- 对外接口 -------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_n: int | None = None,
    ) -> list[Document]:
        """重排并回填 ``rerank_score``；分数越低越不相关，按分数降序返回。"""
        text = (query or "").strip()
        items = [document for document in documents if document.page_content.strip()]
        if not text or not items:
            return []

        limit = _limit(top_n, len(items))
        payload: dict[str, Any] = {
            "model": self.settings.rerank_model,
            "input": {
                "query": text,
                "documents": [document.page_content for document in items],
            },
            "parameters": {"top_n": limit},
        }
        if self.settings.rerank_instruct:
            # 仅 qwen3 系列 rerank 支持；官方建议用英文写任务说明。
            payload["parameters"]["instruct"] = self.settings.rerank_instruct

        data = self._post(payload)
        ranked = _parse_results(data, len(items))

        threshold = self.settings.rerank_min_score
        ordered: list[Document] = []
        for index, score in ranked:
            if threshold is not None and score < threshold:
                continue
            document = items[index]
            # 保留 RRF 分便于排查，同时把 score 换成精排分（0~1，前端展示更直观）。
            document.metadata["retrieval_score"] = document.metadata.get("score")
            document.metadata["rerank_score"] = score
            document.metadata["score"] = score
            document.metadata["reranked"] = True
            ordered.append(document)
        return ordered[:limit]

    # ---- HTTP -----------------------------------------------------------
    def _post(self, payload: dict[str, Any]) -> Any:
        try:
            response = self.client.post(
                self.endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise RerankError(f"调用精排接口失败（{self.endpoint}）：{exc}") from exc

        if response.status_code != 200:
            raise RerankError(f"精排接口返回 HTTP {response.status_code}：{_brief(response.text)}")

        try:
            data = response.json()
        except ValueError as exc:
            raise RerankError(f"精排接口返回的不是合法 JSON：{_brief(response.text)}") from exc

        # 百炼的失败响应可能是 HTTP 200 + {"code": "...", "message": "..."}。
        code = data.get("code") if isinstance(data, dict) else None
        if code:
            message = data.get("message") or "未知错误"
            raise RerankError(f"精排接口返回错误 {code}：{message}")
        return data


def resolve_endpoint(settings: Settings) -> str:
    """决定 rerank 端点：显式配置 > 从百炼域名推导 > 公共网关。

    ``EMBEDDING_BASE_URL`` 常见写法是
    ``https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1``，
    把路径换成 rerank 的固定路径即可复用同一个业务空间与地域；换成 OpenAI 等
    非百炼域名时不瞎推导，直接用公共网关（Key 仍取 ``DASHSCOPE_API_KEY``）。
    若 Key 属于某个业务空间，而 ``EMBEDDING_BASE_URL`` 又不是百炼域名，
    就必须显式配置 ``RERANK_ENDPOINT``。
    """
    if settings.rerank_endpoint:
        return settings.rerank_endpoint

    base = (settings.embedding_base_url or "").strip()
    if not base:
        return DASHSCOPE_RERANK_ENDPOINT

    if settings.is_dashscope_embedding:
        parts = urlsplit(base)
        return f"{parts.scheme}://{parts.netloc}{RERANK_PATH}"
    return DASHSCOPE_RERANK_ENDPOINT


def get_reranker(settings: Settings | None = None) -> Reranker | None:
    """工厂：未启用 / 未配 Key 时返回 ``None``，调用方据此跳过精排。"""
    cfg = settings or get_settings()
    if not cfg.rerank_enabled:
        return None
    if not cfg.resolved_rerank_api_key:
        logger.info("未配置 RERANK_API_KEY / DASHSCOPE_API_KEY，跳过精排（只用 RRF 排序）")
        return None
    return DashScopeReranker(cfg)


# ---- 解析与工具 -----------------------------------------------------------
def _parse_results(data: Any, total: int) -> list[tuple[int, float]]:
    """把 ``output.results`` 解析成 ``[(原始下标, 分数)]``，按分数降序。

    服务端已排好序，这里仍然重排一次并做三重防御：下标越界、重复下标、
    结构异常。任何一项都不值得让整条链路 500，所以坏掉的条目直接跳过。
    """
    output = data.get("output") if isinstance(data, dict) else None
    raw = output.get("results") if isinstance(output, dict) else None
    if not isinstance(raw, list) or not raw:
        raise RerankError(f"精排结果为空或结构异常：{_brief(str(data))}")

    pairs: list[tuple[int, float]] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if not 0 <= index < total or index in seen:
            continue
        raw_score = item.get("relevance_score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            continue
        seen.add(index)
        pairs.append((index, round(float(raw_score), 6)))

    if not pairs:
        raise RerankError(f"精排结果没有可用的 index / relevance_score：{_brief(str(data))}")
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return pairs


def _limit(top_n: int | None, total: int) -> int:
    if top_n is None:
        return total
    return max(1, min(int(top_n), total))


def _brief(text: str, limit: int = 200) -> str:
    flattened = " ".join(str(text).split())
    return flattened if len(flattened) <= limit else f"{flattened[:limit]}…"
