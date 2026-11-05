"""阿里云 OSS 对接（私域 Bucket）：签名客户端、图片服务与接口模型。

模块划分：

* :mod:`ai_agent_py.alioss.client` —— 基于 httpx 的 OSS V1 签名客户端
  （不依赖官方 SDK，PUT / GET / LIST / DELETE / 预签名 URL）；
* :mod:`ai_agent_py.alioss.service` —— 图片上传校验、对象键规划、OSS 直链、
  列表与删除；
* :mod:`ai_agent_py.alioss.models` —— 接口出入参。
"""

from __future__ import annotations

from ai_agent_py.alioss.client import OssClient, OssListResult, OssObject
from ai_agent_py.alioss.models import (
    OssImageOut,
    OssObjectListOut,
    OssObjectOut,
    OssOverviewOut,
    OssPresignOut,
)
from ai_agent_py.alioss.service import OssImageService

__all__ = [
    "OssClient",
    "OssImageOut",
    "OssImageService",
    "OssListResult",
    "OssObject",
    "OssObjectListOut",
    "OssObjectOut",
    "OssOverviewOut",
    "OssPresignOut",
]
