# ai-agent-py 项目说明

## 这是什么

ai-agent-py 是一个用 Python 编写的 RAG（检索增强生成）示例项目，核心框架是 LangChain 与 LangGraph，
依赖安装与虚拟环境统一由 uv 管理。它同时提供命令行工具和 HTTP 接口。

## 技术栈

- 语言：Python，要求 3.11 及以上，推荐 3.13。
- 包管理与版本管理：uv。常用命令是 `uv sync`、`uv add`、`uv run`、`uv lock`。
- 组件框架：LangChain 负责模型、提示词模板、文本切分、检索器与向量库适配。
- 流程编排：LangGraph 负责智能体的状态图，包括 agent 节点、tools 节点和条件边。
- 向量库：默认使用 Chroma 做本地持久化，测试与离线演示可以切换到内存向量库。
- 文档解析：支持 txt、md、pdf、docx 四种格式。
- 命令行：Typer 加 Rich，提供 ingest、ask、chat、demo、info、serve 六个子命令。
- HTTP 服务：FastAPI，提供 /query、/chat、/health、/info 四个接口。

## 目录结构

- src/ai_agent_py：应用代码，包含 config、llm、rag、graph、tools、api、cli 等模块。
- src/ai_agent_py/rag：加载、切分、向量化、检索与 RAG 链。
- src/ai_agent_py/graph：LangGraph 的状态定义、节点与图构建。
- data/raw：待索引的原始文档。
- data/chroma：Chroma 的持久化目录，可以随时删除重建。
- tests：pytest 测试，使用假的模型和向量，不依赖网络。

## 环境变量

关键配置写在项目根目录的 .env 文件中，例如 LLM_MODEL、LLM_API_KEY、LLM_BASE_URL、
EMBEDDING_MODEL、EMBEDDING_API_KEY、VECTOR_STORE、CHUNK_SIZE、RETRIEVER_K。
所有配置都可以用环境变量覆盖，字段说明见 .env.example。

## 常见问题

问：没有 API Key 能跑吗？
答：可以，执行 `ai-agent-py demo` 会用词法哈希向量和占位模型跑通完整流程，只是回答质量不代表真实效果。

问：换一个模型厂商要改哪里？
答：只改 .env 里的 LLM_BASE_URL 和 LLM_MODEL。凡是兼容 OpenAI 接口的服务都可以直接使用。

问：向量库里的内容过时了怎么办？
答：重新执行 `ai-agent-py ingest --recreate`，会清空集合后重新写入。
