"""MinerU 精准解析 API 对接（客户端 / 结果解包 / 任务调度）。

入口：

* :class:`ai_agent_py.mineru.client.MineruClient` — 异步 httpx 客户端；
* :class:`ai_agent_py.mineru.service.DocumentService` — 上传、轮询、落盘、入库；
* :class:`ai_agent_py.mineru.models.ParseJob` — 任务模型（API 响应 / 持久化）。
"""
