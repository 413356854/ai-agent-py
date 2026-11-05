"""Project specific exceptions."""

from __future__ import annotations


class AiAgentError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(AiAgentError):
    """Missing or invalid configuration (usually a missing API key)."""


class ToolError(AiAgentError):
    """A tool call was rejected or failed in an expected way.

    Tool wrappers turn it into a ``错误：...`` string for the model, so a wrong
    path, a bad regex or a missing file never breaks the agent loop.
    """


class DocumentLoadError(AiAgentError):
    """A file could not be loaded or its type is not supported."""


class MineruError(AiAgentError):
    """MinerU API 调用失败（网络、参数、服务端错误）。"""


class MineruAuthError(MineruError):
    """Token 缺失、无效或过期。"""


class MineruTaskFailed(MineruError):
    """云端解析任务返回 `state=failed`。"""


class DatabaseError(AiAgentError):
    """MySQL 连接、认证或查询失败。"""


class MilvusError(AiAgentError):
    """Milvus 连接、建表、写入或检索失败。"""


class MilvusSchemaError(MilvusError):
    """线上集合与代码里的 schema 不一致（字段、维度、内置函数）。"""


class RerankError(AiAgentError):
    """精排模型调用失败（网络、鉴权、返回结构异常）。

    检索链路会捕获它并退回 RRF 排序，所以它出现时只降精度、不中断问答。
    """


class SplitCacheMiss(AiAgentError):
    """切分结果缓存不存在或已过期（服务重启 / 换实例 / 超过 TTL）。"""


class OssError(AiAgentError):
    """阿里云 OSS 调用失败（网络、鉴权、服务端错误）。"""


class OssNotConfigured(OssError):
    """OSS 或访问口令未配置（缺 Bucket / Endpoint / AccessKey / OSS_ACCESS_TOKEN）。"""


class OssAccessDenied(OssError):
    """OSS 侧拒绝访问：AccessKey 无权限、Bucket 策略不允许等。"""


class OssObjectNotFound(OssError):
    """OSS 对象不存在。"""


class OssInvalidKey(OssError):
    """对象键不合法，或不在 OSS_PREFIX 目录内。"""


class OssUploadRejected(OssError):
    """上传内容不符合要求（后缀、大小、文件头校验）。"""

