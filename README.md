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
     └─ 智能体（LangGraph）： agent ⇄ tools(search_knowledge_base) → 答案
```

## 特性

- **两套调用路径，按场景选择**
  - RAG 链（LCEL）：检索一次即生成，延迟低、成本可控，适合问答接口。
  - LangGraph 智能体：`agent ⇄ tools` 状态图，模型自己决定要不要检索、检索几轮。
- **可插拔**：对话模型、向量模型、向量库、文件类型都通过配置或依赖注入替换，业务代码不动。
- **可测试**：`tests/` 全部使用脚本化假模型 + 词法哈希向量，`pytest` 不联网、不花钱。
- **三个入口**：命令行（Typer + Rich）、Python API、HTTP 服务（FastAPI）。
- **中文友好**：分块规则针对中文标点调优，提示词与 CLI 输出均为中文。

## 目录结构

```text
ai-agent-py/
├── pyproject.toml            # 依赖、入口点、ruff/mypy/pytest 配置
├── uv.lock                   # uv 锁定文件（建议提交）
├── .python-version           # 3.13，uv 会自动使用/下载
├── .env.example              # 配置模板：复制为 .env 后填写
├── data/
│   ├── raw/                  # 待索引文档（内置 sample.md）
│   └── chroma/               # Chroma 持久化目录（可随时删除重建）
├── src/ai_agent_py/
│   ├── config.py             # pydantic-settings：全部配置项 + 脱敏描述
│   ├── llm.py                # 对话模型 / 向量模型工厂（OpenAI 兼容协议）
│   ├── offline.py            # 离线用的占位模型与哈希向量（demo & 测试）
│   ├── cli.py                # 命令行：info/ingest/ask/chat/demo/serve
│   ├── rag/
│   │   ├── loaders.py        # txt/md/pdf/docx → Document（含编码回退、页码）
│   │   ├── splitter.py       # 中文分块（按标点递归切分）
│   │   ├── store.py          # 向量库：chroma / memory
│   │   ├── ingest.py         # 建索引流水线（稳定 id、分批、进度回调）
│   │   ├── retriever.py      # similarity / mmr / score_threshold 检索器
│   │   └── chain.py          # RAG 提示词、LCEL 链、答案与引用
│   ├── graph/
│   │   ├── state.py          # AgentState（messages + add_messages 归并）
│   │   ├── nodes.py          # agent 节点、tools 节点、条件路由
│   │   └── agent.py          # 组装并编译 StateGraph
│   ├── tools/__init__.py     # 智能体可用工具（检索工具，可扩展）
│   └── api/app.py            # FastAPI：/query /chat /health /info
└── tests/                    # pytest：加载、切分、链路、图、API、CLI
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
Copy-Item .env.example .env      # 然后编辑 .env，至少填 LLM_API_KEY 与 EMBEDDING_API_KEY

# 4. 建索引 + 提问
uv run ai-agent-py ingest data/raw --recreate
uv run ai-agent-py ask "这个项目用什么管理依赖？"
```

bash / macOS 下把 `Copy-Item` 换成 `cp` 即可。

### 常用命令

| 命令 | 说明 |
| --- | --- |
| `uv run ai-agent-py demo` | 离线跑通全流程，验证代码与数据路径 |
| `uv run ai-agent-py ingest [路径...] --recreate` | 建/重建索引，默认读取 `DATA_DIR` |
| `uv run ai-agent-py ask "问题"` | 单次问答（RAG 链），输出答案 + 引用片段 |
| `uv run ai-agent-py ask "问题" --stream` | 流式输出（逐 token 打印） |
| `uv run ai-agent-py chat` | 多轮对话（LangGraph 智能体，带工具调用与上下文） |
| `uv run ai-agent-py info` | 查看当前配置（密钥脱敏）与索引 chunk 数 |
| `uv run ai-agent-py serve` | 启动 FastAPI，文档见 <http://127.0.0.1:8000/docs> |

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
| GET | `/health` | 版本、模型、向量库与 chunk 数 |
| GET | `/info` | 配置快照（密钥脱敏） |

## 配置

所有配置集中在 `.env`（或真实环境变量，优先级更高），字段定义见 `src/ai_agent_py/config.py`。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL` | `gpt-4o-mini` / 空 / 空 | 对话模型，任何 OpenAI 兼容端点 |
| `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` | `text-embedding-3-small` / 空 / 空 | 向量模型，**必须与建索引时一致** |
| `VECTOR_STORE` | `chroma` | `chroma`（持久化）或 `memory`（进程内，测试用） |
| `COLLECTION_NAME` / `CHROMA_DIR` | `rag_docs` / `data/chroma` | 集合名与持久化目录 |
| `DATA_DIR` | `data/raw` | `ingest` 的默认来源目录 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | 分块大小与重叠（字符数） |
| `RETRIEVER_K` | `4` | 召回条数 |
| `RETRIEVER_SEARCH_TYPE` | `similarity` | `similarity` / `mmr` / `similarity_score_threshold` |
| `AGENT_MAX_TURNS` | `8` | 智能体单轮最多工具调用轮数 |

常见厂商组合（写进 `.env` 即可）：

```ini
# DeepSeek（注意：它没有 embedding 接口，向量模型需另配）
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 硅基流动的向量模型
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3

# 本地 Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=ollama
```

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

## 扩展指南

| 想做的事 | 改哪里 |
| --- | --- |
| 接入新模型厂商 | `.env` 里的 `*_BASE_URL` / `*_MODEL` |
| 换向量库（FAISS、pgvector…） | `rag/store.py::build_vector_store` 加分支 |
| 支持新文件类型（HTML、Excel…） | `rag/loaders.py::load_file` 加分支 |
| 加召回后重排（rerank） | `rag/retriever.py` 外包一层，或在 `chain.py` 前处理 |
| 给智能体加工具 | `tools/__init__.py`，返回 `BaseTool` 即可 |
| 改提示词 | `rag/chain.py::SYSTEM_PROMPT`、`graph/agent.py::AGENT_SYSTEM_PROMPT` |

## 开发

```powershell
uv run pytest            # 30 个测试，全部离线
uv run ruff check .      # 静态检查
uv run ruff format .     # 格式化
uv run mypy              # 类型检查（src 目录）
```

- 代码按 `src/` 布局，包名 `ai_agent_py`，入口点见 `pyproject.toml` 的 `[project.scripts]`。
- 依赖统一用 `uv add` / `uv add --dev` 管理，避免手改 `pyproject.toml` 后忘记锁文件。
- 测试不需要真实模型：`tests/fakes.py` 提供可按脚本返回 `tool_calls` 的假模型，
  `offline.LexicalHashEmbeddings` 提供确定性向量。

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
