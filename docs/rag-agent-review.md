# RAG / Agent 通用实现样例 —— 技术点补充评审

> 评审对象：`src/ai_agent_py/rag/**`（RAG 通用链路）与 `src/ai_agent_py/graph/** + tools/** + reimburse/**`（Agent 通用链路）
> 基线：`pytest` 227 passed（离线）；`ruff check .` 12 条；`mypy` 1 条
> 结论先行：**工程完整度高、降级与测试做得好**，但作为「通用实现样例」还缺 5 类能力：
> **① 检索治理（过滤/版本/上下文预算）② 引用可信度 ③ 评测与可观测 ④ 会话与状态治理 ⑤ Agent 的成本/权限/HITL 边界**。

---

## 一、总体判断

| 维度 | 现状 | 评价 |
| --- | --- | --- |
| 检索链路 | 稠密 + BM25 + RRF + 精排，缺 Key 逐级降级 | 完整，超出一般样例 |
| 生成链路 | 提示词约束引用编号 + 越界编号校正 | 有特色，但只做了「编号合法性」 |
| 入库链路 | 稳定 chunk_id、覆盖写、分批、`split_id` 缓存 | 完整，缺删除对称与增量 |
| Agent | ReAct + 工具循环 + recursion_limit；报销审核 9 节点图（并行+汇合+分支） | 结构清晰，缺预算、持久化、HITL |
| 工程质量 | 227 测试、DI 注入、错误中文化、mypy/ruff 配置 | 好，但门禁当前是红的 |
| 缺位 | 评测、Tracing、鉴权、Token 预算、会话存储 | **通用样例的共性短板** |

---

## 二、RAG 侧待补技术点

### R-01【P0】检索过滤 / 多租户 / 知识库隔离

- **现状**：`MilvusChunkSearcher.search(query, *, k)`（`rag/milvus_search.py:123`）没有过滤参数，`_hybrid` / `_dense` / `_bm25` 也都不接受 `filter`；`MILVUS_TENANT` / `MILVUS_KB` / `MILVUS_NUM_PARTITIONS`（`config.py:86-88`）只参与正整数校验，全仓没有第二处读取；`docs/milvus/create_rag_chunk.py:18` 也写明 `tenant_id` / `kb_id` 被移除。
- **影响**：无法「只在这 3 篇文档里找」、无法按部门/租户/文档类型/生效日期过滤；所有文档混在一个集合里互相污染。
- **补什么**：
  1. `search(query, *, k=None, filter_expr: str = "")`，透传到 `client.hybrid_search/search(..., filter=...)`；
  2. `build_search_filter(doc_ids, doc_type, effective_before, tenant)` 做白名单字段拼装，**禁止把用户输入直接拼进表达式**（现有 `_escape_literal` 只处理字面量转义，防不住语义注入）；
  3. schema 增列 `tenant_id`（多租户用 partition key，需重建集合）、`kb_id`、`doc_type`、`effective_date`、`status`。
- **验收**：假 Milvus 客户端断言 `filter` 被原样下传；接口层 `/api/rag/chat` 支持 `doc_ids` 过滤。

### R-02【P0】法规版本 / 时效过滤（本项目业务强相关）

- **现状**：语料里同一部法存在多版本（`公司法_2023`、司法解释 2014/2020），`rag_chunk` 没有 `effective_date` / `status`，旧版与新版一起召回。
- **影响**：答案可能引用已废止条文，这在税务/法务场景是**合规事故**，比"检索不到"更严重。
- **补什么**：入库写 `effective_date`、`status(in_force/repealed)`、`version`；检索默认 `status == "in_force"`；prompt 头部标注「本批资料生效日期区间」；答案中冲突时提示存在多版本。

### R-03【P0】上下文预算与拼装

- **现状**：`format_documents()`（`rag/chain.py:165`）把召回全文拼接，无 token 计数、无截断；唯一控制是 `RETRIEVER_K=4`。
- **补什么**：独立 `ContextBuilder`：
  - token 预算（模型上下文 × 0.5，预留输出），按精排分从高到低装入，装不下则截断该条（保留句首+句尾）；
  - 去重（cosine > 0.95 或 simhash）；
  - 邻块扩展（见 R-04）；
  - 资料用 `<document index="n" source="...">…</document>` 包裹，配合 R-10 的注入防护。
- **验收**：注入超长 chunk 时总 token 不超预算且顺序不变。

### R-04【P1】邻块 / 父子块（small-to-big）

- **现状**：`chunk_index` 存在（`rag/records.py` 排序在用），检索侧完全没用；法条被切断后「第 X 条……但书」丢半句。
- **补什么**：`expand_neighbors(doc_id, indices, window=1)`，`client.query(filter="doc_id == x and chunk_index in [...]")` 回拼；或入库时写 `parent_id`（Milvus 2.6 `add_field`）实现 small-to-big。

### R-05【P0】引用校验与可追溯

- **现状**：只有 `normalize_citations()`（`rag/chain.py:103`）修「`[43]` 越界编号」，**不复核这句话是否真被第 n 条支持**。
- **补什么**：
  1. 答案后处理：解析 `[n]` → 存在性校验 → 支持度校验（数字/术语重叠，或小 NLI 模型）→ 不通过则降级「未标注来源」或重生成；
  2. 数字归一化匹配（`60%` / `百分之六十` / `0.6`；`5‰` / `千分之五`），这是法条问答最容易错的地方；
  3. 返回 `citation_spans`（chunk 内起止字符），前端正文 `[n]` 可点击跳转/高亮（当前 `RagChatPage.tsx` 只在底部列 chips，正文编号不可点）。

### R-06【P0】入库治理

- **现状**：`MilvusRagWriter.index_document()`（`rag/milvus.py:169`）同步全量 embed → delete → upsert；`_delete_document()` 是私有方法，`DocumentService.delete()`（`mineru/service.py:366`）**不清理 `rag_chunk`**。
- **影响**：删除解析任务后，向量仍在，问答继续引用「已删除」的文档；大批量入库失败要整篇重来，embedding 重复付费。
- **补什么**：
  1. 公开 `delete_document(doc_id)` 并接入删除流程（含 `_document_jobs` 展示兜底）；
  2. embedding 缓存（`sha1(content) -> vector`，落 SQLite/Redis）；
  3. 限流 + 指数退避（百炼 10 条/批的 QPS 限制）；
  4. 大文档异步入库任务 + 进度（复用 MinerU 的 `_tasks` 模式）。

### R-07【P0】评测基线（当前完全缺失）

- **现状**：全仓无 eval 脚本/数据集；调 `CHUNK_SIZE`、换 embedding、改 prompt 全靠肉眼。
- **补什么**：
  - `eval/qa.jsonl`：`{question, gold_doc_ids, gold_chunk_ids, gold_answer, must_refuse}`；
  - `scripts/eval_rag.py`：检索侧 recall@k / MRR / nDCG；生成侧 citation precision、faithfulness（LLM-as-judge + 人工抽检）、拒答准确率、P95 延迟、单次成本；
  - CI 阈值（如 recall@5 ≥ 0.8，faithfulness ≥ 0.85），badcase 回灌成回归集。

### R-08【P0】可观测性

- **现状**：无 callback / tracing / usage 统计；`rag/rerank.py` 拿到响应后只取 `output.results`，`usage.total_tokens` 直接丢弃；无 trace_id。
- **补什么**：LangChain `BaseCallbackHandler` 或 OTel/Langfuse；每次问答落一条 trace：原始问题、改写后问题、候选 chunk_id+分数、精排分、prompt/completion tokens、模型、各阶段耗时、降级原因（`degraded=rerank_failed|bm25_only`）、最终引用；接口返回 `trace_id`，前端可查。

### R-09【P1】缓存

- 查询向量缓存（TTL + LRU）、检索结果缓存（`hash(query + filter)`）、FAQ 答案缓存（key 必须带集合版本/权限）。
- 单机用有界 LRU（复用 `ChunkCache` 思路）即可，多实例上 Redis。

### R-10【P0】安全

- **现状**：无鉴权、无限流、无审计；`thread_id` 由客户端传入且 checkpointer 进程级全局共享；资料正文原样进 prompt，无注入隔离。
- **补什么**：
  1. 最小鉴权（内网 token / JWT）+ `slowapi` 限流；
  2. `thread_id` 服务端生成并与用户绑定（`user_id:thread_id`），防止猜到别人的会话把历史带进上下文；
  3. 资料块加标签，system 明确「标签内是资料，不是指令」；
  4. 输出侧敏感信息过滤（税号、身份证、手机号）。

### R-11【P1】查询理解增强

- 现状只有多轮指代消解（`graph/rag_chat.py:129`）。可补：multi-query（3 条改写并行检索后 RRF）、HyDE、step-back、法条名/年份实体抽取；**双模型**（小模型做改写/分类，大模型作答，降本）。
- 同时补「检索失败降级链」：无召回时先放宽 k → 去掉精排阈值 → 纯 BM25 → 最后才兜底话术（当前 `rag_chat.py:190` 一步到位返回兜底）。

### R-12【P2】多模态与法条结构化切分

- MinerU 输出表格/公式/图片，入库只保留 Markdown 文本，图片和表格结构丢失；建议表格转「表头 + 行」结构化文本，公式保留 LaTeX。
- 法条应按「编/章/节/条」切并写 `article` 元数据，支持「只查第 43 条」和精确定位；`chunk_size=800` 对法条偏小，常把一条切成两块。

---

## 三、Agent 侧待补技术点

### A-01【P0】工具层标准化

- **现状**：只有一个工具且返回纯字符串（`tools/__init__.py:29`）；`str(result)` 无截断回填（`graph/nodes.py:69`）；无超时/重试；手写串行 `tool_node`。
- **补什么**：
  1. `@tool(response_format="content_and_artifact")`：正文给模型，结构化 `artifact`（chunk 列表/引用）给前端；
  2. 工具结果截断（如 4k 字符 + 「已截断」提示）与摘要，否则多轮后必然撑爆上下文；
  3. 换用 LangGraph 预置 `ToolNode`（并行执行 + 错误处理 + `InjectedState`），或给自研节点加并发/超时；
  4. 统一 `args_schema`（Pydantic）校验，模型传错参数时返回可纠正的错误文本；
  5. MCP 接入（`langchain-mcp-adapters`），让工具真正可插拔 —— 这比"在 `tools/__init__.py` 加函数"更符合"通用样例"。

### A-02【P0】循环预算与终止

- **现状**：只有 `recursion_limit = max_turns*2+1`（`graph/agent.py:96`）；无工具调用次数/token/费用/耗时上限；无重复调用检测；超限时 `ask_agent()` 反向找 AI 消息，找不到就把最后一条（可能是 `ToolMessage`）当答案返回（`agent.py:127-133`）。
- **补什么**：
  - State 增 `steps / tool_calls / tokens`（`Annotated[int, operator.add]`）；
  - `route_after_agent` 先判预算，超限走显式 `finalize` 节点（提示词：「基于已获得的资料作答，并说明信息可能不完整」）；
  - `(tool_name, args_hash)` 去重，同一查询不重复检索；
  - `asyncio.timeout` 包裹单步。

### A-03【P0】会话持久化与记忆

- **现状**：全项目 `InMemorySaver`（`api/app.py:104`、`api/rag.py:73`、`cli.py:385`），进程重启即丢、多 worker 不共享、无 TTL/上限（长时间运行内存只增不减）；Agent 侧历史全量进上下文，无压缩。
- **补什么**：
  1. `langgraph-checkpoint-sqlite` / Postgres saver；
  2. 会话元数据表 + `/api/threads`（列表/删除/重命名/导出）；
  3. 历史摘要压缩：超过 N 轮把早期消息换成 LLM 摘要 + 保留最近 K 条；
  4. 可选长期记忆（用户偏好、常用文档）。

### A-04【P1】Human-in-the-loop

- **现状**：`reimburse` 的 G8「人工复核」（`reimburse/graph.py:217`）只是返回 `status="转人工"`，没有真正暂停/恢复；外部审批无法回填。
- **补什么**：`interrupt({"type": "approval", "payload": ...})` + `Command(resume=...)`；前提是持久化 checkpointer；配套 `POST /api/agents/{id}/runs/{run_id}/resume`、超时/转交/批量审批。这正是报销场景的刚需，比再加规则更有价值。

### A-05【P1】流式与中间事件

- **现状**：`/chat` 一次性返回；只有 RAG 有 SSE。
- **补什么**：`graph.astream(stream_mode=["messages", "updates", "custom"])` + `get_stream_writer()` 发 `tool_start` / `tool_result` / `thought` 事件；前端时间轴展示「检索中 → 找到 3 段 → 生成中」；断线重连用 `Last-Event-ID`。

### A-06【P2】编排模式

- 单 ReAct + 固定链之外，通用样例应演示：**路由**（意图分类：闲聊/知识库/工具/转人工）、**planner-executor**（复杂问题拆解）、**reflection**（生成后自检引用再修订）、**subgraph 复用**（报销审核作为工具/子图被主 Agent 调用）。
- `reimburse` 的 `extract` 节点目前是启发式（`reimburse/graph.py:118` 注释「启发式可换 LLM」），可换成 `with_structured_output(Pydantic)` 演示结构化抽取与置信度回填。

### A-07【P1】模型可靠性

- 无 fallback 模型（主模型 5xx 直接 502）、无单步超时、`LLM_MAX_TOKENS` 默认 `None`（答案长度不设限）、`bind_tools` 对不支持 function calling 的模型无降级。
- 补：模型注册表（primary / fallback / small）、`max_tokens`、结构化输出解析失败重试、tool_call 解析失败时把文本当最终答案。

### A-08【P0】评测（trajectory 级）

- 指标：任务成功率、工具选择正确率、平均步数、token 成本、转人工率、幻觉率。
- 做法：用 `reimburse` 的 7 个样例做 golden set；多轮场景用「模拟用户」跑对话；把每次失败的轨迹（消息序列）存下来做回归。

### A-09【P0】安全与权限

- 工具 allowlist + 参数校验 + 危险工具需审批（与 A-04 共用中断机制）；工具输出注入隔离（`<tool_result>` 包裹）；per-user 会话隔离；工具调用审计日志（谁、何时、调了什么、返回什么）。

---

## 四、两个样例共用的工程质量缺口

| 编号 | 问题 | 证据 | 建议 |
| --- | --- | --- | --- |
| C-01 | **双索引不一致**：同一份解析结果有两条入库路径——`POST /api/documents/{id}/ingest` 写 Chroma（`mineru/service.py` 的 `ingest`），`POST /api/documents/{id}/rag` 写 Milvus（`service.py:495`）；而 Agent 工具与 `/query` 只查 Chroma（`rag/retriever.py`），RAG 问答只查 Milvus | 同一份资料可能只在一套索引里；两套检索策略也不同（similarity vs hybrid+RRF+rerank），同一问题两个入口答案口径不一致 | 统一到已有的 `DocumentSearcher` Protocol，Agent 工具也注入 Milvus searcher；或在 UI 上明确标出两个知识库，并在两套索引间做状态一致性校验 |
| C-02 | 重复实现：`_add_node`（`graph/agent.py:50`、`graph/rag_chat.py:225`）、`NO_CONTEXT_ANSWER`（`rag/chain.py:21`、`graph/rag_chat.py:52`）、`_snippet`、`_escape_literal`/`_quote`（`rag/records.py:271`、`rag/milvus.py:284`）、`_as_int`/`_optional_int` | — | 抽 `common/` 模块 |
| C-03 | 配置注入不一致：`agent_config()` 读全局 `get_settings()`（`graph/agent.py:94`），忽略 `build_agent_graph(settings=...)` 注入值 | 测试/多租户场景会串配置 | 让 `ask_agent` 接收设置或从 graph 上取 |
| C-04 | **质量门禁是红的**：`ruff check .` 12 条（全部集中在 `reimburse/**` 与 `api/agents.py`）、`mypy` 1 条（`reimburse/graph.py:318`），而 README 的「开发」一节把这两条列为质量门禁 | 见评审基线 | 修 lint（`io.py:288` 的 B023 目前因同步调用不构成真实 bug，但建议用默认参数绑定消除告警） |
| C-05 | 测试缺口：无并发/超时/重试/提示注入/预算耗尽/checkpointer 持久化/SSE 断连测试 | `tests/` 227 条以正常路径为主 | 补 6~8 条异常路径用例 |
| C-06 | 前端：正文 `[n]` 不可点击（`RagChatPage.tsx` 只在底部列 chips）；无 👍/👎 反馈 | 缺少评测数据来源 | 引用锚点 + 反馈落库，作为 R-07 的数据源 |

---

## 五、代码级缺陷（可直接修）

| 编号 | 位置 | 问题 | 修法 |
| --- | --- | --- | --- |
| D-01 | `graph/rag_chat.py:196` | 历史消息原样带进 prompt，**上一轮答案里的 `[1] [2]` 与本轮重新从 1 开始的资料编号冲突**；`normalize_citations` 又因 `n <= total` 判定合法而不修正 → 引用漂移 | 渲染历史时剥离/重映射引用，或历史只保留用户问题 + 摘要 |
| D-02 | `mineru/service.py:366` + `rag/milvus.py:235` | 删除任务不清 `rag_chunk`，孤儿向量继续被检索 | 公开 `delete_document()` 并在删除流程调用 |
| D-03 | `rag/ingest.py:55` + `rag/splitter.py:162` | `chunk_index` 是多文件**全局**序号，插入/删除文件后所有 `chunk_id` 变化 → 「重灌幂等」失效（Milvus 路径按文档编号，不受影响；Chroma 路径受影响） | 按 `source` 分组编号，或 id 只取「source + 文档内序号 + 内容 hash」 |
| D-04 | `api/app.py:104`、`api/rag.py:73` | `InMemorySaver` 无 TTL/上限，任意 `thread_id` 可无限增长内存 | 见 A-03 |
| D-05 | `graph/agent.py:127-133` | 达到 recursion_limit 时最后一条可能是 `ToolMessage`，会把工具原文当答案返回 | 显式判断 `message.type == "ai"`，否则走 finalize 文案 |
| D-06 | `rag/milvus_search.py:173` | 精排失败静默降级，接口未透出 `degraded`，前端仍显示「已精排」 | metadata 加 `degraded_reason`，API/前端展示 |
| D-07 | `rag/milvus_search.py:96` | `_prepare()` 30s 缓存：集合重建（维度/analyzer 变更）后短暂用旧判断 | 缓存里带 `schema_version`/describe 摘要比对 |
| D-08 | `api/rag.py:187/207` | 同步 `def` + 同步 `graph.stream`，靠线程池扛并发 | 改 `astream` + `StreamingResponse` 异步生成器 |
| D-09 | `rag/records.py:188` | 无关键词时扫 16k 行到 Python 排序，集合变大后列表接口慢 | 让 `doc_id`+`chunk_index` 参与服务端排序（或落 MySQL 索引表） |
| D-10 | `rag/chain.py:165` | 资料无边界标签，正文里的「忽略以上指令」会和提示词同层 | 见 R-10 |

---

## 六、建议落地顺序

**第 1 周（P0，止血）**
D-01 / D-02 / D-05 缺陷修复 → R-01 过滤 + R-02 版本 → R-05 引用校验 → A-02 预算 → R-10 最小鉴权。

**第 2 周（P0，可度量）**
R-03 上下文预算 + R-04 邻块 → R-07 评测集与脚本 → R-08 tracing/usage → R-06 删除对称与缓存 → A-01 工具标准化 → A-03 持久化 saver。

**第 3 周（P1）**
A-04 HITL → A-05 Agent SSE → A-08 轨迹评测 → R-09 缓存 → R-11 查询增强 → C-04 修门禁。

**后续（P2）**
C-01 统一检索入口 → A-06 多模式编排 → R-12 多模态与法条结构化切分 → C-06 反馈闭环。

---

## 附：两个最小骨架

```python
# 1) 过滤贯通（rag/milvus_search.py）
def search(self, query: str, *, k: int | None = None, filter_expr: str = "") -> list[Document]:
    ...
    hits = self._hybrid(text, vector, fetch, candidates, filter_expr)
    ...

def _hybrid(self, text, vector, limit, candidates, filter_expr=""):
    result = self.client.hybrid_search(
        self.collection, requests,
        ranker=RRFRanker(self.settings.milvus_rrf_k),
        limit=limit, output_fields=list(SEARCH_FIELDS),
        filter=filter_expr or None,          # ← 唯一改动点，所有检索共用
    )
```

```python
# 2) Agent 预算（graph/state.py + graph/nodes.py）
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: Annotated[int, operator.add]
    tool_calls_used: Annotated[int, operator.add]

def route_after_agent(state: AgentState, budget: int = 8) -> Literal["tools", "finalize", "end"]:
    last = state["messages"][-1]
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return "end"
    if state.get("tool_calls_used", 0) >= budget:
        return "finalize"          # 用已有资料作答，而不是直接抛 recursion_limit
    return "tools"
```
