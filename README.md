# ai-agent-py

基于 **LangChain + LangGraph** 的 RAG（检索增强生成）项目骨架，用 **uv** 管理 Python 版本与依赖。

开箱即用：`uv sync` 之后可以直接 `uv run ai-agent-py demo`，**不需要任何 API Key** 就能离线跑通
「加载 → 切块 → 向量化 → 检索 → 生成」全流程；配置好 `.env` 后即可切换成真实模型。

```text
┌──────────┐   loaders   ┌──────────┐  splitter  ┌────────┐  embeddings  ┌────────┐
│ txt/md   │ ──────────▶ │ Document │ ─────────▶ │ chunks │ ───────────▶ │ Chroma │
│ pdf/docx │             └──────────┘            └────────┘              └────┬───┘
└──────────┘                                                                 │
                                        ┌───────────────────────────────────┘
   问题 ──────────────────────────────▶ retriever ──▶ 相关片段（带来源）
     │
     ├─ RAG 链（LCEL）：  prompt(上下文 + 问题) → LLM → 答案 + 引用
     ├─ 智能体（LangGraph）： agent ⇄ tools(search_knowledge_base) → 答案
     └─ RAG 问答（LangGraph）： condense → retrieve(Milvus rag_chunk) → answer
                                → SSE 流式答案 + 引用片段
```

## 特性

- **三套调用路径，按场景选择**
  - RAG 链（LCEL）：检索一次即生成，延迟低、成本可控，适合问答接口。
  - LangGraph 智能体：`agent ⇄ tools` 状态图，模型自己决定要不要检索、检索几轮。
  - RAG 问答（LangGraph 固定链）：`condense → retrieve(Milvus) → answer`，多轮改写 + 混合检索 +
    精排 + 引用，前端 SSE 流式渲染。
- **可插拔**：对话模型、向量模型、向量库、文件类型都通过配置或依赖注入替换，业务代码不动。
- **结构化切分 + 缓存式入库 RAG**：切分统一读解析结果 `markdown_file`（JSON 直读任务从
  `full.md` 的代码块还原原文后按结构切），切片在服务端 HashMap 里缓存并返回 `split_id`；
  前端「切分」弹窗把切片逐条展示给用户查阅，点「入库 RAG」时只回传 `split_id`
  （请求体几十字节），后端从缓存取完整切片写入 Milvus `rag_chunk`
  （百炼 `text-embedding-v4`，1536 维），不重新切分、不受预览截断影响。
  **入库成功后文档记录显示「已入库」**；**允许重复入库**——每次先删除该文档的旧切片，
  再写入新切片，换块大小或正文有改动都不会留下旧块 / 重复块。
- **Milvus 建表脚本**：通用业务 RAG 表（稠密 + BM25 + 分区过滤 + 引用元数据），
  一条命令建集合，见 [docs/milvus/create_rag_chunk.py](docs/milvus/create_rag_chunk.py)。
- **MySQL 对接**：直接连接本地 `db_py_agent`（JDBC 风格 URL 自动翻译）。
  文档解析的任务记录落在 `tb_document_parse_job`（建表脚本见 [docs/mysql/](docs/mysql/)），
  另提供 `ai-agent-py db check` 与 `/api/db/health` 的连接/库信息检查。
- **可测试**：`tests/` 全部使用脚本化假模型 + 词法哈希向量，`pytest` 不联网、不花钱。
- **三个入口**：命令行（Typer + Rich）、Python API、HTTP 服务（FastAPI）。
- **文档解析**：对接 [MinerU 精准解析 API](https://mineru.net/apiManage/docs)（推荐 `vlm` 模型），
  上传 PDF / Word / PPT / 图片 → 云端解析 → Markdown（含表格、公式、图片），可一键写入知识库。
- **管理后台**：React 19 + TypeScript + Ant Design，栏目「RAG 管理 → 文档解析 / RAG 记录 / RAG 问答」，
  支持多文件、多批次加入待解析队列，点击后按顺序逐个提交，任务进度、Markdown 预览、下载、
  切分预览、入库 RAG 一站式完成；右上角一键切换 **亮色 / 暗色主题**。
- **RAG 记录**：直接分页浏览 Milvus `rag_chunk`，可按文档 / 正文关键词（中缀）过滤，展示总块数、
  文档分布与向量维度；文档名按 `doc_id` 从 MySQL 任务表回查，一键跳到「文档解析」看原文。
  列表先按 `(doc_id, chunk_index)` 全局排序再回查正文，翻页时块序号连续、可读；内容只展示前两行，
  **点击整格可在弹窗查看完整片段**（含文档 ID、块 ID、页码、字数）并一键复制全文。
- **RAG 问答**：聊天式问答页 + **LangGraph** 编排的 `condense → retrieve → answer` 图：
  多轮追问先改写成独立问题，再在 `rag_chunk` 上做稠密 + BM25 + RRF 混合检索，RRF 后再用
  `qwen3.7-text-rerank` 精排，最后带 `[1] [2]` 引用作答；答案通过 **SSE 逐 token 推送**，
  引用片段可点开查看原文与相关度；页头标签实时显示 LLM 模型、检索方式与精排是否启用。
  （没有 Embedding Key 时自动退化为 BM25 关键词检索；没有精排 Key 时跳过精排；没有 LLM Key
  时页面给出明确提示。）
- **Agent 应用（报销审核）**：`reimburse` 是一张 9 节点的 LangGraph 工序图（G1 接入 → G2 抽取 →
  G3/G4/G5 三类检查并行 → G6 判定融合 → G7 结论 / G8 人工 → G9 归档），规则库 13 条（YAML 元数据 +
  判定器注册表）。后台「Agent 应用」栏目直接展示 **`get_graph().draw_mermaid()` 导出的流程图**，
  可切换主题、查看原始 Mermaid 源码；试跑页支持**下载填充好的 xlsx 样板 / 空白模板**，
  也可以**直接上传填好的单据（xlsx / json）**解析后试跑；语义疑点只会转人工，不会自动退回。
- **中文友好**：分块规则针对中文标点调优，提示词与 CLI / 前端输出均为中文。

## 目录结构

```text
ai-agent-py/
├── pyproject.toml            # 依赖、入口点、ruff/mypy/pytest 配置
├── uv.lock                   # uv 锁定文件（建议提交）
├── .python-version           # 3.13，uv 会自动使用/下载
├── .env.example              # 配置模板：复制为 .env 后填写
├── scripts/
│   ├── milvus-start.ps1      # 启动 WSL1 里的 Milvus（独立 wsl.exe 宿主 + healthz 就绪等待）
│   └── milvus-stop.ps1       # 停止 Milvus（SIGTERM，等待 19530 端口关闭）
├── data/
│   ├── raw/                  # 待索引文档（内置 sample.md）
│   ├── parsed/               # MinerU 解析结果（Markdown + 图片；任务记录在 MySQL）
│   └── chroma/               # Chroma 持久化目录（可随时删除重建）
├── src/ai_agent_py/
│   ├── config.py             # pydantic-settings：全部配置项 + 脱敏描述
│   ├── llm.py                # 对话模型 / 向量模型工厂（OpenAI 兼容协议）
│   ├── offline.py            # 离线用的占位模型与哈希向量（demo & 测试）
│   ├── cli.py                # 命令行：info/db/ingest/ask/chat/demo/parse/serve
│   ├── db/
│   │   ├── dsn.py            # JDBC mysql:// → mysql+pymysql://（字符集 / 时区 / 超时）
│   │   ├── health.py         # 连接池 + 只读健康检查（版本 / 库 / 表数 / 连接池）
│   │   └── document_jobs.py  # tb_document_parse_job 的行映射与 upsert / 查询 / 删除
│   ├── rag/
│   │   ├── loaders.py        # txt/md/pdf/docx → Document（含编码回退、页码）
│   │   ├── splitter.py       # 按后缀：Markdown / HTML / JSON / 递归切分
│   │   ├── chunk_cache.py    # 切分结果缓存：split_id → 完整切片（TTL + LRU）
│   │   ├── milvus.py         # 切分块写入 Milvus rag_chunk（稠密 + BM25，覆盖写）
│   │   ├── milvus_search.py  # rag_chunk 混合检索（稠密 + BM25 + RRF + 精排，缺 Key 退 BM25）
│   │   ├── rerank.py         # RRF 之后精排：qwen3.7-text-rerank（DashScope 原生接口）
│   │   ├── records.py        # rag_chunk 只读查询（RAG 记录列表 / 文档分布）
│   │   ├── models.py         # 切分预览 / RAG 入库 / RAG 记录 / 问答的接口模型
│   │   ├── store.py          # 向量库：chroma / memory
│   │   ├── ingest.py         # 建索引流水线（稳定 id、分批、进度回调）
│   │   ├── retriever.py      # similarity / mmr / score_threshold 检索器
│   │   └── chain.py          # RAG 提示词、LCEL 链、答案与引用
│   ├── mineru/               # MinerU 精准解析 API 对接
│   │   ├── client.py         # 异步 httpx：申请上传链接 / PUT / 轮询 / 下载
│   │   ├── service.py        # 任务调度：上传落盘、进度、结果持久化、入库
│   │   ├── archive.py        # 结果 zip 解包（防 zip slip、图片、页数）
│   │   └── models.py         # ParseJob / ParseOptions / 状态机
│   ├── graph/
│   │   ├── state.py          # AgentState（messages + add_messages 归并）
│   │   ├── nodes.py          # agent 节点、tools 节点、条件路由
│   │   ├── agent.py          # 组装并编译 StateGraph
│   │   └── rag_chat.py       # RAG 问答图：condense → retrieve → answer
│   ├── tools/__init__.py     # 智能体可用工具（检索工具，可扩展）
│   ├── reimburse/            # 报销审核 Agent：规则库 + 9 节点图 + 图导出
│   │   ├── rules.py          # 13 条规则（YAML 元数据 + 判定器注册表）
│   │   ├── params.py         # 参数库存储：内存缓存 + 原子落盘 + 失效重载
│   │   ├── io.py             # 单据模板导出/导入（xlsx 双向 + json）
│   │   ├── graph.py          # G1–G9 工序图，describe_graph() 导出 mermaid
│   │   ├── samples.py        # 7 张合成样例单据（通过/退回/转人工）
│   │   └── data/             # rules.yaml + policy_params.yaml（占位参数）
│   └── api/
│       ├── app.py            # FastAPI：/query /chat /health /info + 前端托管
│       ├── database.py       # MySQL 健康检查接口：/api/db/health
│       ├── documents.py      # 文档解析接口：上传 / 状态 / 预览 / 下载 / 切分 / 入库
│       ├── agents.py         # Agent 应用：列表 / 图导出 / 样例 / 试跑
│       └── rag.py            # RAG 记录列表 + RAG 问答（含 SSE 流式）
├── web/                      # 管理后台（文档解析 / RAG 记录 / RAG 问答 / Agent 应用）
└── tests/                    # pytest：加载、切分、链路、图、API、CLI、MinerU、MySQL、RAG 记录
```

## 快速开始

前置：安装 [uv](https://docs.astral.sh/uv/)（Windows 可用 `winget install --id=astral-sh.uv` 或
`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`）。

```powershell
# 1. 安装依赖（自动创建 .venv，并按 .python-version 选择 Python 3.13）
uv sync

# 2. 离线自检：不需要 API Key
uv run ai-agent-py demo

# 3. 配置真实模型
Copy-Item .env.example .env      # 然后编辑 .env：填 LLM_API_KEY（或导出 DASHSCOPE_API_KEY）与向量模型 Key

# 4. 检查本地 MySQL（库已存在时只做连接检查）
uv run ai-agent-py db check

# 5. 建索引 + 提问
uv run ai-agent-py ingest data/raw --recreate
uv run ai-agent-py ask "这个项目用什么管理依赖？"

# 6. 管理后台（构建一次，之后由 FastAPI 直接托管）
cd web
pnpm install
pnpm build
cd ..
uv run ai-agent-py serve          # 打开 http://127.0.0.1:8000/
```

bash / macOS 下把 `Copy-Item` 换成 `cp` 即可。前端需要 Node.js 20+ 与 pnpm。

### 常用命令

| 命令 | 说明 |
| --- | --- |
| `uv run ai-agent-py demo` | 离线跑通全流程，验证代码与数据路径 |
| `uv run ai-agent-py ingest [路径...] --recreate` | 建/重建索引，默认读取 `DATA_DIR` |
| `uv run ai-agent-py ask "问题"` | 单次问答（RAG 链），输出答案 + 引用片段 |
| `uv run ai-agent-py ask "问题" --stream` | 流式输出（逐 token 打印） |
| `uv run ai-agent-py chat` | 多轮对话（LangGraph 智能体，带工具调用与上下文） |
| `uv run ai-agent-py parse 文件.pdf -o out.md` | 调用 MinerU 精准解析 API 把文档转成 Markdown |
| `uv run ai-agent-py db check` | 连接本地 MySQL，输出服务端版本、库名、表数、耗时与连接池 |
| `uv run ai-agent-py info` | 查看当前配置（密钥脱敏）与索引 chunk 数 |
| `uv run ai-agent-py serve` | 启动 FastAPI + 管理后台，<http://127.0.0.1:8000/> |

也可以 `uv run python -m ai_agent_py ...`，行为完全一致。

### HTTP 接口

```powershell
uv run ai-agent-py serve --port 8000

Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post `
  -ContentType 'application/json' -Body '{"question":"这个项目用什么管理依赖？","k":4}'
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/query` | RAG 链问答，返回 `answer` + `sources`（来源、页码、片段） |
| POST | `/chat` | 智能体对话，`thread_id` 相同则保留上下文 |
| GET | `/api/agents` | Agent 应用列表（节点数 / 规则数） |
| GET | `/api/agents/{id}/graph` | LangGraph 图导出：`mermaid` 源码 + 节点 / 边 |
| GET | `/api/agents/{id}/config` | 规则库与参数（参数默认标 `placeholder`） |
| PUT | `/api/agents/{id}/config/params` | 覆盖写入参数：校验 → 落盘 → 刷新缓存，立即生效 |
| POST | `/api/agents/{id}/config/reload` | 手工改了 YAML 后重新读取参数文件 |
| POST | `/api/agents/{id}/run` | 试跑一单：`{"sample_id": "C001"}` 或 `{"claim": {...}}` |
| GET | `/api/agents/{id}/template` | 空白报销单模板（xlsx） |
| GET | `/api/agents/{id}/samples/{sid}/download` | 样例样板下载（`fmt=xlsx` 或 `json`） |
| POST | `/api/agents/{id}/upload` | 上传填好的单据（xlsx/json）→ 解析成 claim |
| GET | `/health` | 版本、模型、向量库、chunk 数与 MySQL 连接状态 |
| GET | `/info` | 配置快照（密钥脱敏） |
| GET | `/api/db/health` | MySQL 连接检查：服务端版本、库名、表数、时区、耗时、连接池 |
| GET | `/api/documents/config` | 解析能力、可上传后缀、RAG 入库目标（集合 / 维度 / 是否配置 Key） |
| POST | `/api/documents/parse` | 上传文档（multipart）→ 立即返回任务（202）；`.md` / `.json` 直接完成 |
| GET | `/api/documents` | 任务列表（`state` / `keyword` / `limit` 过滤） |
| GET | `/api/documents/{id}` | 任务状态与进度 |
| GET | `/api/documents/{id}/markdown` | Markdown 预览 |
| GET | `/api/documents/{id}/download?kind=markdown\|zip\|source` | 下载结果 |
| GET | `/api/documents/{id}/chunks` | **切分预览**：读 `markdown_file` 切分，返回 `split_id` + 片段与统计 |
| POST | `/api/documents/{id}/rag` | **按 `split_id` 从缓存取切片写入 Milvus** `db_tax_rag.rag_chunk`（可重复入库：先删旧切片；成功后任务记录 `rag_ingested=true`） |
| DELETE | `/api/documents/{id}` | 删除任务、结果与源文件 |
| POST | `/api/documents/{id}/ingest` | 把解析结果写入 Chroma（`VECTOR_STORE`） |
| GET | `/parsed/{id}/images/...` | 解析结果里的图片（Markdown 预览用） |
| GET | `/api/rag/overview` | **RAG 记录 / 问答**概览：Milvus 连接、总块数、文档分布、模型配置状态（含精排 `rerank_model` / `rerank_configured`） |
| GET | `/api/rag/chunks` | **`rag_chunk` 记录列表**：`doc_id` / `keyword` / `page` / `page_size` |
| POST | `/api/rag/chat` | **RAG 问答**（LangGraph，`thread_id` 保留多轮上下文）→ `answer` + `sources` |
| POST | `/api/rag/chat/stream` | 同上，SSE 流式：`stage` → `sources` → `token`… → `done` |

## 切分与入库 RAG（百炼 text-embedding-v4 + Milvus）

「文档解析」列表每行都有 **切分** 按钮，流程是 **切分 → 缓存 → 用户查阅 → 入库**：

1. 后端**只读**任务的 `markdown_file`（MinerU 产出的 `full.md`；`.md` 任务就是源文件）
   切分，保证弹窗里的切片与 Markdown 预览同源；
2. 切分结果写进服务端进程内 HashMap（`rag/chunk_cache.py`），接口返回 `split_id`；
   弹窗逐条展示切片（标题层级、页码、字符数），改块大小后点「重新切分」会生成新 id；
3. 点 **入库 RAG** 时，前端只 `POST {"split_id": "...", "total": N}`，后端按 id 从
   缓存取出**完整**切片（预览截断不影响入库）写入 `db_tax_rag.rag_chunk`，返回
   `mode="cached"`。不传 `split_id` 的调用方仍可回传 `chunks`（无状态 / 多实例），
   或两者都不传让服务端重新切分（CLI / 脚本）。

| 文档类型 | 切分来源 | 切分器 | 说明 |
| --- | --- | --- | --- |
| `.md` / `.markdown` | 源文件（即 `markdown_file`） | `MarkdownHeaderTextSplitter` | 按 `#`~`####` 标题层级切，标题写入 metadata |
| `.json` | `full.md` 里的 json 代码块 | `RecursiveJsonSplitter` | 取回块内原文按嵌套结构切；单个超长字符串值不再硬切，保证片段是合法 JSON |
| HTML / PDF / DOCX / PPT / 图片 | MinerU 产出的 `full.md` | `MarkdownHeaderTextSplitter` | 不再回读原始 HTML；统一按 Markdown 标题结构切 |

- 结构化切分器只决定「在哪里切」，不保证块大小；除 JSON 外都会再走一遍递归切分兜底，
  确保单块 ≤ `chunk_size`。
- `.md` / `.json` 不调用 MinerU，上传后立即完成。
- **缓存边界**：默认 TTL 30 分钟、最多 16 条、总字符 8M（LRU 淘汰）；只存在本进程内存，
  服务重启 / 换实例后 `split_id` 失效，接口返回 409 并提示重新切分（前端会自动重切）。
  多实例部署要么做粘性会话，要么用无状态的 `chunks` 入参。
- 弹窗最多渲染 5000 个切片；超出时只提示「仅展示前 N 个」，入库仍会写入缓存的全部切片。
- 前端回传时带上 `total`（切分总数）：与实际入库数量不一致直接 422，避免切分结果错位。
- 删除任务时会清掉该任务的全部切分缓存。
- **重复入库**：每次写 Milvus 前先按 `doc_id` 删除该文档的**全部**旧切片，再 `upsert`
  本次切片；换 `chunk_size` / 修改正文后重新入库不会留下孤儿块或重复块。embedding
  计算成功后才执行删除，向量模型失败不会先丢掉旧数据。
- **已入库状态**：RAG 入库成功后任务记录写入独立的 `rag_ingest`（块数、字符数、目标集合、
  入库时间），列表「状态」列在解析状态下方显示「已入库」；重新入库只刷新这份摘要，
  可以反复操作。它与 Chroma 入库的 `ingest` 分开，两者互不覆盖。
- 向量模型默认走阿里云百炼 OpenAI 兼容接口：`DASHSCOPE_API_KEY` +
  `text-embedding-v4` + `EMBEDDING_DIMENSIONS=1536`（必须等于 `MILVUS_DIM`）。
- `sparse`（BM25）由 Milvus 服务端根据 `content` 生成，应用写入时不赋值。

```powershell
# 1. 确认 Milvus 与集合已就绪（dim 与 EMBEDDING_DIMENSIONS 一致）
uv run python docs/milvus/create_database.py
uv run python docs/milvus/create_rag_chunk.py --dim 1536

# 2. 设置百炼 Key（推荐环境变量；也可写进 .env）
$env:DASHSCOPE_API_KEY = "sk-..."

# 3. 启动管理后台：上传 → 切分 → 查阅切片 → 入库 RAG
uv run ai-agent-py serve --port 8000
```

## RAG 记录与 RAG 问答（Milvus `rag_chunk` + LangGraph）

入库完成后，管理后台多了两个栏目：**RAG 记录**（核对写进 `rag_chunk` 的内容）和
**RAG 问答**（聊天式检索问答）。两者都只读 Milvus 的 `rag_chunk`，与「文档解析 → 切分 →
入库 RAG」共用同一份数据。

![RAG 记录](docs/screenshots/rag-records.png)

![RAG 记录片段详情](docs/screenshots/rag-record-detail.png)

![RAG 问答（亮色）](docs/screenshots/rag-chat.png)

![RAG 问答（暗色）](docs/screenshots/rag-chat-dark.png)

### RAG 记录（`GET /api/rag/chunks`）

- 只返回 `chunk_id` / `doc_id` / `chunk_index` / `page` / `content`，**不搬** `embedding` /
  `sparse` 两个向量字段；
- 总数用 Milvus `count(*)` 聚合，分页与过滤严格一致；
- 过滤条件：`doc_id`（精确）+ `keyword`（`content like "%kw%"` 中缀匹配，Milvus 2.5+ 支持）；
- Milvus `query` **不支持 ORDER BY**，所以常规浏览会先扫一遍轻量元数据（主键 + 文档 +
  序号 + 页码），在应用侧按 `(doc_id, chunk_index)` 排序，再按本页主键回查正文——翻页时
  块序号连续，读起来就是原文顺序；
- 关键词查询 / 集合超过 16384 块时退回服务端 `offset/limit` 分页（只保证页内有序，
  响应里的 `truncated=true` 会提示）；
- 文档名不在 Milvus 里：接口用 `doc_id` 回查 MySQL `tb_document_parse_job`，查不到就显示
  `doc_id`，MySQL 不可用不影响记录浏览；
- 列表的「内容」列只展示前两行，**点击整格弹出片段详情**：展示完整正文（可滚动）、文档 ID /
  块 ID / 块序号 / 页码 / 字数，并支持「复制全文」；表头带「点击查看全文」提示，键盘
  `Enter` / 空格也可打开详情。

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/rag/overview"
Invoke-RestMethod "http://127.0.0.1:8000/api/rag/chunks?keyword=业务招待费&page=1&page_size=10"
```

### RAG 问答（`graph/rag_chat.py`，LangGraph）

```text
START ──▶ condense ──▶ retrieve ──▶ answer ──▶ END
            改写追问      Milvus 检索    带引用作答
```

- **condense**：首轮直接使用用户问题，**不额外调模型**；存在历史对话时，才让 LLM 把
  「那它的扣除标准呢？」改写成可独立检索的问题（提示词 `CONDENSE_PROMPT`）；
- **retrieve**：`MilvusChunkSearcher` 在 `rag_chunk` 上做 **稠密 + BM25 + RRF（+ 精排）** 混合检索，
  两路各取 `MILVUS_CANDIDATES` 条，用 `RRFRanker(MILVUS_RRF_K)` 融合；配了精排时保留
  `RERANK_CANDIDATES` 条候选交给 `qwen3.7-text-rerank` 重新打分，最后取 `RETRIEVER_K` 条。
  精排失败（超时 / 401 / 配额）只记 warning 并退回 RRF 顺序，不影响问答可用性；没配精排 Key
  时这一步直接跳过（等价于只用 RRF）；
- **answer**：把片段按 `[1] [2]` 编号拼进提示词，并声明「共 N 条，编号 1~N」；**没有召回时
  不调用 LLM**，直接返回兜底文案（省钱、避免幻觉）；
- **引用校正**：模型偶尔把法条序号当来源编号（「第四十三条」写成 `[43]`）。提示词里明确禁止并
  给了反例，`rag/chain.py::normalize_citations` 再做一层确定性兜底：若恰好一条片段正文包含
  「第 N 条」就改写成该片段的编号（`[43]` → `[2]`），无法唯一定位则写成「（第四十三条）」；
  越界数字会打 warning 日志，不影响有效编号；
- 多轮上下文由 LangGraph `InMemorySaver` 按 `thread_id` 保存在**服务端进程内**：刷新页面后
  点「新会话」换一个 `thread_id` 即可重新开始；服务重启后旧会话失效；
- 前端用 `POST /api/rag/chat/stream`（SSE）逐 token 渲染，事件顺序为
  `stage`（检索中）→ `sources`（引用片段）→ 若干 `token` → `done`（完整答案，以它为准）；
  出错时发 `error` 事件；
- 降级：没有 `EMBEDDING_API_KEY` 时只走 BM25 关键词检索（集合本来就带 analyzer / BM25，
  `content` 就是索引字段）；连 `LLM_API_KEY` / `DASHSCOPE_API_KEY` 都没有时页面会提示
  「问答不可用」，接口返回 503。
- 页头标签来自 `GET /api/rag/overview`：LLM 模型、`混合检索（向量 + BM25）`/`BM25 检索`、
  `精排 qwen3.7-text-rerank`/`未启用精排`；没开精排时页面上方还有一条提示，说明会送多少条
  候选、最终取几条。引用弹窗里的「相关度」在精排后是 0~1 的精排分。

```powershell
# 非流式（脚本 / 联调）
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/rag/chat -Method Post `
  -ContentType 'application/json' `
  -Body '{"message":"企业所得税的扣除项目有哪些？","thread_id":"demo-1"}'

# 流式（SSE）：浏览器里由前端 fetch 读取，命令行可用 curl -N
curl.exe -N -X POST http://127.0.0.1:8000/api/rag/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"message":"业务招待费的扣除标准是什么？","thread_id":"demo-1"}'
```

## Milvus 通用业务 RAG 表

默认链路走 Chroma，够用；需要**生产级混合检索**（稠密语义 + BM25 关键词 + RRF 融合 +
qwen3.7 精排 + 页码引用）时，用 Milvus 存检索块。脚本在 `docs/milvus/`：

- [`create_database.py`](docs/milvus/create_database.py)：建库 `db_tax_rag`（幂等，可设副本/资源组/配额）；
- [`create_rag_chunk.py`](docs/milvus/create_rag_chunk.py)：建表 `rag_chunk`，**7 字段 + 3 索引 + BM25 函数**。

```powershell
uv sync                                                       # pymilvus 已是主依赖
uv run python docs/milvus/create_database.py                  # 建库 db_tax_rag
uv run python docs/milvus/create_rag_chunk.py --dim 1536      # 建表（默认进 db_tax_rag）
uv run python docs/milvus/create_rag_chunk.py --print-spec    # 不连库，打印字段/索引 JSON
uv run python docs/milvus/create_database.py --drop           # 删库（要求库内为空）
```

### 本机 Milvus（WSL1）启停

本机 Milvus 2.6 以 **原生 DEB 解包**方式装在 WSL1 `Ubuntu-20.04` 的 `~/milvus`（不用 Docker、
不用 systemd），`scripts/` 里带了一份启停脚本：

```powershell
# 启动：独立 wsl.exe 宿主常驻，等到 healthz 返回 200 才退出；已在运行则直接返回
pwsh -File scripts/milvus-start.ps1

# 停止：发 SIGTERM，等待 19530 端口关闭
pwsh -File scripts/milvus-stop.ps1

# 看日志
wsl -d Ubuntu-20.04 -- tail -f /home/ljf/milvus/milvus.log
```

- 就绪检查：<http://127.0.0.1:9091/healthz>（返回 200 即可用）；WebUI：<http://127.0.0.1:9091/webui/>
- 默认 WSL 发行版 `Ubuntu-20.04`、安装目录 `/home/ljf/milvus`，可用
  `-Distro Ubuntu-20.04` / `-WslHome /home/<用户>/milvus` / `-TimeoutSeconds 300` 覆盖；
- 启动脚本用 `Start-Process wsl.exe` 拉起独立宿主，关掉当前终端后 Milvus 仍继续运行；
- 脚本只覆盖**启停**；Milvus 本体需要先按原安装流程装到 WSL 里（本机已装好）。

建表常用参数：`--uri` / `--token`（或环境变量 `MILVUS_URI` / `MILVUS_TOKEN`）、`--database`
（默认 `db_tax_rag`，写 `default` 回默认库）、`--collection`、`--dim`（必须等于 embedding
模型输出维度）、`--metric`、`--analyzer`（`chinese` / `english` / `standard`，建表后不可改）、
`--recreate`。

表设计要点：

- 一行 = 一个检索块，只有 7 个字段：`chunk_id` / `doc_id` / `chunk_index` / `page` /
  `content` / `embedding` / `sparse`；
- **文档级属性不进 Milvus**：文件名、后缀、页数、时间在 MySQL `tb_document_parse_job`，
  `doc_id` 就是该表的 `id`，检索后按它回查拼引用（避免改标题要更新 N 行）；
- 一个集合两路向量：`embedding`（稠密）+ `sparse`（BM25，服务端自动生成，**不要手写**）；
- 主键是稳定 `chunk_id`（`sha1(doc_id|chunk_index|content)`），重灌幂等，不会产生重复块；
- `content` 开 analyzer 后同时作为 BM25 输入；`page` 用于引用回溯（PDF 有，docx 转 md 为 NULL）；
- 将来要加过滤维度（租户/知识库）或分区键，再按需 `add_field` 或重建集合；
- 换 embedding 模型 = 换维度 = **建新集合 + 回填 + 别名切换**，不要原地改；
- 集合在 Milvus 重启后是 `released` 状态，检索前要 `load_collection`（脚本创建时已 load）。

应用侧写入在 `src/ai_agent_py/rag/milvus.py`（`MilvusRagWriter`）：按 `doc_id` 覆盖写、
只写 `content` + `embedding`（`sparse` 由服务端生成）、删除旧块后 `upsert` 并 flush。
读取在 `rag/records.py`（记录列表），混合检索（`embedding` + `sparse` + RRF）在
`rag/milvus_search.py`，已接入「RAG 问答」（`/api/rag/chat`）；`/query` 仍走 Chroma 链路。

## MySQL 对接（只读健康检查）

本次对接的库是 `db_py_agent`。只做**连接与健康检查**：
打开连接池、执行 `SELECT` 读取服务端信息，**不建表、不改数据、不读取业务表内容**。
如果这个库还没有建表，`db check` 会显示表数量 0，但连接本身仍算正常。

```ini
MYSQL_URL=jdbc:mysql://localhost:3306/db_py_agent?useUnicode=true&characterEncoding=utf8&connectionTimeZone=Asia/Shanghai&allowPublicKeyRetrieval=true&useSSL=false
MYSQL_USERNAME=admin
MYSQL_PASSWORD=change-me
MYSQL_POOL_SIZE=5
MYSQL_MAX_OVERFLOW=10
MYSQL_POOL_RECYCLE=1800
MYSQL_CONNECT_TIMEOUT=5
```

`MYSQL_URL` 同时接受 Java 的 `jdbc:mysql://` 与 SQLAlchemy 的 `mysql+pymysql://` 写法，
连接参数在 `src/ai_agent_py/db/dsn.py` 里翻译：

| 原参数 | 处理 |
| --- | --- |
| `characterEncoding=utf8` | → `charset=utf8mb4`（中文与 emoji 都能存） |
| `connectionTimeZone=Asia/Shanghai` | → 连接时 `SET time_zone = '+08:00'` |
| `connectTimeout`（JDBC，毫秒） | 由 `MYSQL_CONNECT_TIMEOUT`（秒）统一控制 |
| `useUnicode` / `allowPublicKeyRetrieval` / `useSSL=false` | JDBC 专用，PyMySQL 默认行为已满足 |

```powershell
uv run ai-agent-py db check
# 或
Invoke-RestMethod http://127.0.0.1:8000/api/db/health
```

输出包括：状态、服务端版本、当前账号、库名、表数量、会话时区、连接耗时、连接池
（`size` / `checkedin` / `checkedout`）。失败时 `status=down` 并给出中文原因（密码错误、
库不存在、服务未启动、超时等），CLI 以退出码 1 结束。密码只存在 `.env` 与引擎内部，
`/info`、`/api/db/health` 和日志里的连接串都显示为 `***`。

> 连接池使用 `pool_pre_ping=True`：MySQL 重启或空闲连接被回收后，下一次检查会自动重连，
> 不会把失效连接留给请求。健康检查失败不会影响 RAG / 文档解析等现有功能。

### 建库脚本

数据库编码读取结果（`information_schema.SCHEMATA`）：

| 库 | 默认字符集 | 默认排序规则 |
| --- | --- | --- |
| `db_py_agent`（读取时） | `utf8mb4` | `utf8mb4_general_ci` |
| `db_ai_agent`（源库，23 张表） | `utf8mb4` | `utf8mb4_0900_ai_ci` |
| 服务端默认 | `utf8mb4` | `utf8mb4_0900_ai_ci` |

[`docs/db/create_database.sql`](docs/db/create_database.sql) 默认复刻 `db_py_agent`
读取到的值（`utf8mb4` + `utf8mb4_general_ci`），可在 MySQL 客户端中直接执行；脚本只创建 / 对齐
数据库本身，不建表、不写数据。文末另附一段取消注释即可使用的 `ALTER`，用于把默认排序规则
改成与源库一致的 `utf8mb4_0900_ai_ci`。

## 文档解析（MinerU 精准解析 API）

对接官方 [精准解析 API](https://mineru.net/apiManage/docs)：支持 `pipeline` / `vlm`（默认，精度最高）/
`MinerU-HTML`，单文件 ≤ 200 MB、≤ 600 页，支持 PDF、图片、Doc/Docx、Ppt/PPTx。

![文档解析列表（亮色）](docs/screenshots/documents-list.png)

![文档解析列表（暗色）](docs/screenshots/documents-dark.png)

![Markdown 预览（暗色）](docs/screenshots/documents-dark-preview.png)

主题切换按钮在页面右上角（月亮/太阳图标）：选择写入 `localStorage`，刷新后保持；
从未切换过时跟随系统 `prefers-color-scheme`。`index.html` 里的内联脚本会在首屏渲染前
写好 `<html data-theme>`，避免暗色模式下闪白。

### 为什么走「批量上传」接口

精准解析 API 的单文件接口 `POST /extract/task` 只接受 **URL**，本地文件必须走批量上传链路：

```text
① POST /file-urls/batch            申请上传链接 → batch_id + file_urls
② PUT  <file_url>                  上传文件字节（不设 Content-Type）
                                   ← MinerU 自动提交解析任务
③ GET  /extract-results/batch/{id} 轮询 state: pending → running → done
④ GET  <full_zip_url>              下载 zip：full.md + images/ + content_list.json
```

`ai_agent_py.mineru.MineruClient` 封装了 ①②③④，`DocumentService` 负责调度：

```text
浏览器 ──上传──▶ POST /api/documents/parse ──▶ 落盘 data/parsed/<job>/source.pdf
                                                    │  + INSERT tb_document_parse_job
                                                    │ 后台 asyncio 任务
                                                    ▼
                         MinerUClient：申请链接 → PUT → 轮询 → 下载 zip
                                                    │
                        解包 → full.md / images/ → 每次状态变化 UPDATE 该行
                                                    │
浏览器 ──轮询──▶ GET /api/documents/{id} ◀──── 状态：uploading → parsing → done
         └──────▶ GET /api/documents/{id}/markdown（预览，图片走 /parsed/<job>/）
         └──────▶ POST /api/documents/{id}/ingest（切块写入 Chroma）
```

- **异步任务**：`POST /parse` 立刻返回，前端每 2.5s 轮询一次，不占用 HTTP 连接。
- **网络重试**：上传、轮询、下载遇到连接超时 / TLS 抖动（例如代理导致的
  `SSLV3_ALERT_BAD_RECORD_MAC`）/ 5xx 时自动重试 3 次（1s、2s 退避）；创建任务的
  POST 不重试，避免重复建任务。
- **断点续跑**：任务记录写在 MySQL 表 `tb_document_parse_job`（一行一个任务），
  服务启动后从库里恢复未完成的任务继续轮询。
- **多实例可见**：列表、详情、预览、下载、入库都直接读库，任意实例都能看到别的实例创建的任务；
  只有本进程正在解析的任务走内存对象（页数进度比库里更新），避免进度回退。
- **结果落盘**：`data/parsed/<job_id>/`（`full.md`、`images/`、`mineru.zip`、`source.*`），
  删除任务会一并删除该目录。`.md` / `.json` 不走 MinerU，上传后立刻标记完成。
- **两种入库**：行内「入库」把 Markdown 写进 Chroma（`COLLECTION_NAME`）；「切分 →
  查阅 → 入库 RAG」把服务端缓存的切片写进 Milvus `db_tax_rag.rag_chunk`。前者缺
  `EMBEDDING_API_KEY`、后者缺 `DASHSCOPE_API_KEY` / `EMBEDDING_API_KEY` 时都返回 503 并给出提示。

### 命令行用法

```powershell
uv run ai-agent-py parse "D:\docs\年报.pdf" --model-version vlm -o "data\年报.md"
```

### 直接调 HTTP 接口

```powershell
$job = curl.exe -s -X POST http://127.0.0.1:8000/api/documents/parse `
  -F "file=@demo.pdf" -F "model_version=vlm" -F "language=ch" | ConvertFrom-Json
Invoke-RestMethod "http://127.0.0.1:8000/api/documents/$($job.id)"
```

## 配置

所有配置集中在 `.env`（或真实环境变量，优先级更高），字段定义见 `src/ai_agent_py/config.py`。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL` | `gpt-4o-mini` / 空 / 空 | 对话模型（任何 OpenAI 兼容端点）；`LLM_API_KEY` 留空时回退环境变量 `DASHSCOPE_API_KEY`、再回退 `OPENAI_API_KEY` |
| `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` | `text-embedding-3-small` / 空 / 空 | 向量模型，**必须与建索引时一致** |
| `DASHSCOPE_API_KEY` | 空 | 阿里云百炼 Key；`LLM_API_KEY` / `EMBEDDING_API_KEY` 未填时分别作为对话 / 向量密钥 |
| `EMBEDDING_DIMENSIONS` | 空 | 请求的向量维度（v4 默认 1024；本项目的 Milvus 表是 1536，必须显式指定） |
| `EMBEDDING_BATCH_SIZE` | `64` | 单次 embedding 文本数；百炼兼容接口建议 `10` |
| `VECTOR_STORE` | `chroma` | `chroma`（持久化）或 `memory`（进程内，测试用） |
| `COLLECTION_NAME` / `CHROMA_DIR` | `rag_docs` / `data/chroma` | 集合名与持久化目录 |
| `DATA_DIR` | `data/raw` | `ingest` 的默认来源目录 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | 分块大小与重叠（字符数） |
| `RETRIEVER_K` | `4` | 召回条数 |
| `RETRIEVER_SEARCH_TYPE` | `similarity` | `similarity` / `mmr` / `similarity_score_threshold` |
| `AGENT_MAX_TURNS` | `8` | 智能体单轮最多工具调用轮数 |
| `MINERU_API_KEY` | 空 | MinerU Token（[申请](https://mineru.net/apiManage/token)），文档解析必需 |
| `MINERU_MODEL_VERSION` | `vlm` | `pipeline` / `vlm`（推荐）/ `MinerU-HTML`（仅 HTML） |
| `MINERU_LANGUAGE` | `ch` | OCR 语言，扫描件/图片务必选对 |
| `MINERU_ENABLE_FORMULA` / `MINERU_ENABLE_TABLE` | `true` | 公式 / 表格识别 |
| `MINERU_IS_OCR` | `false` | 扫描件 OCR（pipeline / vlm 生效） |
| `MINERU_POLL_INTERVAL` / `MINERU_TIMEOUT` | `3` / `600` | 轮询间隔与单任务最长等待（秒） |
| `MINERU_MAX_CONCURRENT` / `MINERU_MAX_FILE_MB` | `2` / `200` | 并发解析数与单文件大小上限 |
| `MINERU_OUTPUT_DIR` | `data/parsed` | 解析产物目录（源文件 / Markdown / 图片 / zip；任务记录在 MySQL） |
| `MYSQL_URL` / `MYSQL_USERNAME` / `MYSQL_PASSWORD` | 空 / 空 / 空 | 本地已有库的连接串与账号；支持 `jdbc:mysql://` 写法 |
| `MYSQL_POOL_SIZE` / `MYSQL_MAX_OVERFLOW` | `5` / `10` | 常驻连接数与峰值溢出数 |
| `MYSQL_POOL_RECYCLE` | `1800` | 连接回收秒数（避免 MySQL 侧 `wait_timeout` 断开） |
| `MYSQL_CONNECT_TIMEOUT` | `5` | 建连超时秒数；健康检查失败时也最多等这么久 |
| `MILVUS_URI` / `MILVUS_TOKEN` | `http://localhost:19530` / 空 | Milvus 地址与 token（Zilliz Cloud 填 API Key） |
| `MILVUS_DATABASE` / `MILVUS_COLLECTION` | `db_tax_rag` / `rag_chunk` | 入库目标库与表名 |
| `MILVUS_DIM` / `MILVUS_METRIC` | `1536` / `COSINE` | 必须与 `EMBEDDING_DIMENSIONS` / 模型输出维度一致 |
| `MILVUS_TENANT` / `MILVUS_KB` | `default` / `default` | 租户（分区键）与知识库范围 |
| `MILVUS_ANALYZER` | `chinese` | `chinese`（jieba）/ `english` / `standard`，建表后不可改 |
| `MILVUS_BATCH_SIZE` | `500` | 单次 `upsert` 行数 |
| `MILVUS_CANDIDATES` / `MILVUS_RRF_K` / `MILVUS_EF` | `30` / `60` / `64` | 混合检索参数（详见设计文档） |
| `RERANK_ENABLED` / `RERANK_MODEL` | `true` / `qwen3.7-text-rerank` | RRF 之后的精排开关与模型；没 Key 时自动跳过 |
| `RERANK_API_KEY` | 空 | 留空依次回退 `DASHSCOPE_API_KEY`、百炼域名下的 `EMBEDDING_API_KEY`；都没有则不做精排 |
| `RERANK_CANDIDATES` / `RERANK_MIN_SCORE` | `30` / 空 | 送进精排的候选数与最低分（分数只在单次请求内可比） |

常见厂商组合（写进 `.env` 即可）：

```ini
# DeepSeek（注意：它没有 embedding 接口，向量模型需另配）
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 阿里云百炼 text-embedding-v4（OpenAI 兼容模式，本项目默认）
EMBEDDING_BASE_URL=https://ws-9lonm8753e95w4oc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1536
EMBEDDING_BATCH_SIZE=10
# 密钥优先读环境变量 DASHSCOPE_API_KEY

# 硅基流动的向量模型
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3

# 本地 Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=ollama
```

文档解析还需要 MinerU Token（在 <https://mineru.net/apiManage/token> 申请）：

```ini
MINERU_API_KEY=sk-xxxxxxxxxxxxxxxx
MINERU_MODEL_VERSION=vlm
MINERU_LANGUAGE=ch
```

> Token 只写在 `.env`（已被 `.gitignore` 忽略），接口 `/api/documents/config` 只返回是否已配置，
> `/info` 里的密钥也是脱敏的。

## 架构说明

### 1. RAG 链（`rag/chain.py`）

```python
answer = answer_question("问题", llm=llm, retriever=retriever)  # -> RagAnswer(answer, sources)
chain = build_rag_chain(llm, retriever)  # LCEL: {question} -> {answer, documents}
```

- 提示词把召回片段编号渲染成 `[1] 来源：xxx`，要求模型只依据资料回答并标注编号。
- 没有召回结果时**直接返回兜底答案，不调用 LLM**（省钱且避免幻觉）。
- `format_documents()` 同时被检索工具复用，保证两套路径的引用格式一致。

### 2. LangGraph 智能体（`graph/`）

```text
          ┌──────────────────────────────┐
          ▼                              │
START ──▶ agent ──有 tool_calls？──是──▶ tools
             │
             └──否──▶ END
```

- `agent` 节点：把工具绑定到模型后调用一次，返回的 `AIMessage` 追加进 `messages`。
- `tools` 节点：执行上一条消息里所有 `tool_calls`，把结果作为 `ToolMessage` 回填。
- 条件边 `route_after_agent`：还有工具调用就继续循环，否则结束。
- `recursion_limit` 由 `AGENT_MAX_TURNS` 换算，防止无限循环。
- 需要多轮记忆时传入 `checkpointer=InMemorySaver()`（CLI 的 `chat` 与 API 的 `/chat` 已内置）。

新增工具只需在 `tools/__init__.py` 里加函数，再 `build_agent_graph(tools=[...])`，图本身不用改。

### 3. RAG 问答图（`graph/rag_chat.py` + `rag/milvus_search.py`）

```text
START ──▶ condense ──▶ retrieve ──▶ answer ──▶ END
            │            │            │
         改写追问      rag_chunk     带 [1][2] 引用
        （首轮跳过）   RRF + 精排    无召回不调 LLM
```

- `RagChatState`：`messages`（`add_messages` 追加）+ `query` / `documents` / `answer`
  （每轮覆盖）；`condense` / `retrieve` / `answer` 三个节点都是普通函数，可单独测试。
- 检索分两段：`MilvusChunkSearcher` 先做稠密 + BM25 + RRF 融合，配了 Key 时再把
  `RERANK_CANDIDATES` 条候选交给 `rag/rerank.py`（`qwen3.7-text-rerank`）精排后取
  `RETRIEVER_K` 条；精排异常被吞掉并退回 RRF 顺序，不影响问答可用性。
- `build_rag_chat_graph(llm=..., searcher=..., checkpointer=...)` 注入依赖：测试用
  `ScriptedChatModel` + 假检索器，不联网；生产用 `ChatOpenAI(streaming=True)` +
  `MilvusChunkSearcher`。
- 答案节点把历史消息（最近 6 条）+ 编号资料 + 本轮问题一起交给模型，所以追问可以引用上文，
  但事实依据仍限定在本次检索结果里。
- API 层（`api/rag.py`）把 `graph.stream(..., stream_mode=["messages", "updates"])` 翻译成
  SSE：只转发 `answer` 节点里 `AIMessageChunk` 的 token（`condense` 的改写过程不展示，
  非流式模型产出的完整消息也不重复推送，最终以 `done.answer` 为准）。

### 4. 文档解析（`mineru/` + `api/documents.py`）

| 文件 | 职责 |
| --- | --- |
| `mineru/client.py` | 异步 httpx 客户端，兼容业务信封（`code/msg/data`）与网关信封（`success/msgCode`），错误码翻译成中文提示 |
| `mineru/service.py` | 任务调度：落盘、申请上传链接、轮询、下载、并发上限、MySQL 持久化与重启续跑、读 `markdown_file` 切分并缓存 / 按 `split_id` 入库 |
| `rag/splitter.py` | 按后缀路由 Markdown / HTML / JSON / 递归切分（文档解析链路统一用 Markdown），统一块大小兜底与编号 |
| `rag/chunk_cache.py` | 切分结果缓存：`split_id` → 完整切片，TTL / 条数 / 字符预算与 LRU 淘汰 |
| `rag/models.py` | `ChunkIn` / `ChunkOut` 与 `split_id` 入库参数：前端查阅、回传切片的数据模型 |
| `rag/milvus.py` | `rag_chunk` 写入：稳定主键、覆盖写、embedding 分批、schema / 维度校验 |
| `mineru/archive.py` | 结果 zip 解包：防 zip slip / zip bomb、挑出 `full.md`、统计图片与页数（`content_list.json` 的 `page_idx`） |
| `mineru/models.py` | `ParseJob`（API 响应 + 持久化）、`ParseOptions`、`JobState` 状态机 |
| `db/document_jobs.py` | 任务记录表 `tb_document_parse_job` 的 SQLAlchemy 定义、行映射与 upsert / 查询 / 删除 |
| `api/documents.py` | REST 接口 + 解析结果静态资源（`/parsed/{job}/...`） |

- 任务状态：`queued → uploading → parsing → downloading → done`，任意阶段失败进入 `failed`；
  RAG 入库不改变解析状态，另有 `rag_ingest` 摘要与 `rag_ingested=true` 表示「已入库」。
- 解析结果只保存相对文件名（`full.md`、`mineru.zip`），绝对路径不外泄，静态资源按 `job_id` 校验。
- 前端 `web/` 构建后由 FastAPI 挂在 `/`；未构建时 `/` 会显示一段构建提示，接口照常可用。

#### 任务记录表（MySQL）

应用不会自动改库结构，首次部署要手动执行建表脚本：

```powershell
mysql -u admin -p db_py_agent < docs/mysql/tb_document_parse_job.sql
# PowerShell 没有输入重定向时：
Get-Content docs/mysql/tb_document_parse_job.sql | mysql -u admin -p db_py_agent
```

- 表名 `tb_document_parse_job`，一行一个解析任务；`id` 同时是 `data/parsed/<id>/` 的目录名。
- 原 JSON 里嵌套的 `options`（模型版本 / OCR 语言 / 公式表格开关 / 页码范围）与
  `ingest`（入库摘要）摊平成 `model_version`、`language`、`enable_formula`… 与 `ingest_*` 列，
  可以直接按列筛选统计，不用解析 JSON。RAG（Milvus）入库另记在 `rag_ingest_*` /
  `rag_ingested_at`：非 NULL 即列表里的「已入库」，与 Chroma 的 `ingest_*` 互不覆盖。
- 建表脚本已包含 `rag_*` 四列；已按旧版建过表的库，需要自行补齐这四列。
- 时间列一律存 **UTC**（`DATETIME(6)`），接口输出仍带 `Z`；要展示东八区时间用
  `CONVERT_TZ(created_at, '+00:00', '+08:00')`。
- 旧版 `data/parsed/jobs.json` 可一次性迁入（按主键 upsert，可重复执行）：

```powershell
uv run ai-agent-py db import-jobs --dry-run   # 只解析预览，不写库
uv run ai-agent-py db import-jobs             # 写入 tb_document_parse_job
```

## 扩展指南

| 想做的事 | 改哪里 |
| --- | --- |
| 接入新模型厂商 | `.env` 里的 `*_BASE_URL` / `*_MODEL` |
| 换向量库（FAISS、pgvector…） | `rag/store.py::build_vector_store` 加分支 |
| 支持新文件类型（HTML、Excel…） | `rag/loaders.py::load_file` 加分支 |
| 加召回后重排（rerank） | `rag/rerank.py`（已内置 `qwen3.7-text-rerank`）；Chroma 链路可再包一层 `DocumentCompressor` |
| 调 RAG 问答检索策略 | `rag/milvus_search.py`（两路候选数 `MILVUS_CANDIDATES`、`RRFRanker(MILVUS_RRF_K)`、精排候选 `RERANK_CANDIDATES`、缺 Key 退 BM25） |
| RAG 记录加筛选维度 | `rag/records.py::build_filter` + `rag/models.py::RagChunkListOut` + 前端筛选栏 |
| 解析更多文件类型 | `mineru/service.py::ALLOWED_SUFFIXES` + `.env` 里的 `MINERU_MAX_FILE_MB` |
| 换解析服务（Unstructured 等） | 实现与 `MineruClient` 相同的方法，替换 `DocumentService(client_factory=...)` |
| 管理后台加栏目 | `web/src/layout/AdminLayout.tsx` 的 `MENU_ITEMS` + `web/src/App.tsx` 路由 |

## 开发

```powershell
uv run pytest            # 216 个测试，全部离线（MinerU / 精排用 httpx.MockTransport，任务记录库用内存 JobStore，RAG 记录 / 问答用假 Milvus 客户端与脚本化模型）
uv run ruff check .      # 静态检查
uv run ruff format .     # 格式化
uv run mypy              # 类型检查（src 目录）

cd web
pnpm install
pnpm dev                 # 5173，/api 与 /parsed 代理到 8000
pnpm build               # 产物 web/dist，由 FastAPI 托管
```

- 代码按 `src/` 布局，包名 `ai_agent_py`，入口点见 `pyproject.toml` 的 `[project.scripts]`。
- 依赖统一用 `uv add` / `uv add --dev` 管理，避免手改 `pyproject.toml` 后忘记锁文件。
- 主题：`web/src/theme/ThemeContext.tsx` 管理亮/暗状态（写 `<html data-theme>` 与
  `localStorage`），`main.tsx` 用 antd 的 `darkAlgorithm` 切换组件配色，自定义样式走
  `styles.css` 顶部的 CSS 变量，新增自定义组件时直接用变量即可自动适配两套主题。
- 测试不需要真实模型：`tests/fakes.py` 提供可按脚本返回 `tool_calls` 的假模型 /
  流式假模型（`StreamingScriptedChatModel`，覆盖 SSE 逐 token 推送），
  `offline.LexicalHashEmbeddings` 提供确定性向量，`tests/test_mineru.py` 里的 `FakeMineru`
  用 `httpx.MockTransport` 模拟 MinerU 全流程（上传 / 轮询 / 下载 zip）；
  `tests/test_database.py` 用假引擎覆盖成功与密码错误两条路径，不会连本机 MySQL；
  `tests/test_rag_records.py` / `tests/test_milvus_search.py` / `tests/test_rag_chat_graph.py` /
  `tests/test_rag_api.py` 用假 Milvus 客户端与脚本化模型覆盖 RAG 记录与问答，不连 Milvus。

## 常见问题

**没有 API Key 能跑吗？**
可以。`ai-agent-py demo` 使用哈希向量 + 占位模型，只为验证代码路径与数据是否就绪，回答质量不代表真实效果。

**`uv sync` 卡住或超时？**
国内网络可换镜像：

```powershell
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv sync
```

**换了 embedding 模型后检索结果很差？**
向量维度与语义空间都变了，必须重建索引：`uv run ai-agent-py ingest --recreate`。

**PDF 检索不到内容？**
扫描版 PDF 没有文本层，需要先 OCR。加载器遇到无文本页面会打印警告并跳过。

**DeepSeek 能用吗？**
对话可以，但它不提供 embedding 接口，向量模型要单独配置（硅基流动 / OpenAI / Ollama 等）。

**数据放在哪里？**
默认 `data/raw/`，支持子目录递归；也可以 `uv run ai-agent-py ingest D:\docs --recreate` 指定任意目录。
`data/raw/` 与 `data/chroma/` 已在 `.gitignore` 中忽略。

**文档解析提示「Token 无效 / 未配置」？**
在 `.env` 里填 `MINERU_API_KEY` 后重启服务；页面顶部有黄色提示说明未配置。
官方每天 2000 页高优先级额度，超出后排队优先级降低。

**解析一直是「排队中」或失败？**
`pending/running` 属于云端排队，稍等即可；`failed` 会用中文给出原因（格式不支持、超 200MB、
超 600 页、额度用尽、文件读取失败等）。扫描件请打开「扫描件 OCR」并选对「文档语言」。

**解析结果和任务记录在哪？**
解析产物在 `data/parsed/<job_id>/`（`full.md`、`images/`、`mineru.zip`、源文件）；
任务记录在 MySQL 表 `tb_document_parse_job`（一行一个任务，含状态、参数、进度与入库摘要）。
删除任务会同时删除记录与该目录，服务重启后会继续轮询未完成的任务。

**管理后台打不开？**
先构建一次前端：`cd web && pnpm install && pnpm build`，再重启 `serve`；未构建时打开 `/` 会提示构建命令。
前端仓库里的 `web/dist` 与 `web/node_modules` 都不提交。

**RAG 记录是空的 / 打不开？**
页面顶部会显示具体原因。常见情况：Milvus 没启动（`MILVUS_URI`）、集合还没建
（`docs/milvus/create_rag_chunk.py`）、或者文档还没「入库 RAG」。集合存在但没有数据时，
列表为 0 条是正常的；MySQL 不可用时只影响「文档名」列，记录本身仍能浏览。

**RAG 问答提示「问答不可用 / 未就绪」？**
需要 `LLM_API_KEY`，留空时会自动回退环境变量 `DASHSCOPE_API_KEY`，再回退 `OPENAI_API_KEY`。
`EMBEDDING_API_KEY` 是可选的：没有时检索会自动退化为 BM25 关键词检索（页面会提示），
配上之后才是向量 + 关键词混合检索。精排同理：没有 `RERANK_API_KEY` / `DASHSCOPE_API_KEY`
时自动跳过（只用 RRF 排序），配了之后才走 `qwen3.7-text-rerank`。如果集合里没有数据，
先到「文档解析」切分并入库。

> 只给 `DASHSCOPE_API_KEY` 时记得把对话端点也指向百炼：
> `LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`、`LLM_MODEL=qwen-plus`
> （或你在百炼控制台开通的模型）；否则 Key 会被发到默认的 OpenAI 端点。

**RAG 问答的多轮上下文存在哪里？**
LangGraph 的 `InMemorySaver`，按前端生成的 `thread_id` 存在**服务端进程内**。刷新页面或点
「新会话」会换一个新的 `thread_id`；服务重启后旧会话失效（不会报错，只是从头开始）。
多实例部署时请用粘性会话，或把 checkpointer 换成持久化实现（如 Postgres / Redis）。

**亮色 / 暗色主题怎么切换？**
页面右上角的月亮 / 太阳按钮，选择保存在浏览器 `localStorage`，刷新或重启服务都保持；
首次访问跟随系统 `prefers-color-scheme`。想改默认色板就调 `web/src/styles.css` 里的主题变量。
