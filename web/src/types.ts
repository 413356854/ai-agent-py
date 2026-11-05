/** 与后端 `ai_agent_py.mineru.models` / `ai_agent_py.rag.models` 对应的类型。 */

export type JobState = 'queued' | 'uploading' | 'parsing' | 'downloading' | 'done' | 'failed'

export interface ParseOptions {
  model_version: string
  language: string
  enable_formula: boolean
  enable_table: boolean
  is_ocr: boolean
  page_ranges: string | null
}

/** RAG（Milvus `rag_chunk`）入库摘要。 */
export interface RagIngestInfo {
  chunks: number
  characters: number
  collection: string
  ingested_at: string
}

export interface ParseJob {
  id: string
  file_name: string
  file_size: number
  suffix: string
  state: JobState
  options: ParseOptions
  created_at: string
  updated_at: string
  started_at: string | null
  finished_at: string | null
  elapsed_seconds: number | null
  batch_id: string | null
  err_msg: string
  extracted_pages: number
  total_pages: number
  source_file: string | null
  markdown_file: string | null
  zip_file: string | null
  markdown_chars: number
  asset_count: number
  /** 最近一次 RAG 入库摘要；null 表示未入库 RAG。 */
  rag_ingest: RagIngestInfo | null
  /** 后端计算的「是否已入库 RAG」状态。 */
  rag_ingested: boolean
  is_terminal: boolean
  progress: number
  state_label: string
  assets_base: string
}

export interface DocumentConfig {
  configured: boolean
  base_url: string
  model_version: string
  default_options: ParseOptions
  model_versions: string[]
  languages: string[]
  max_file_mb: number
  accepted_suffixes: string[]
  output_dir: string
  rag_configured: boolean
  embedding_model: string
  rag_database: string
  rag_collection: string
  rag_dim: number
}

export interface MarkdownOut {
  job_id: string
  file_name: string
  markdown: string
  assets_base: string
  characters: number
  assets: number
}

/** 一个切分片段（与后端 `rag.models.ChunkOut` 对应）。 */
export interface ChunkOut {
  index: number
  content: string
  characters: number
  metadata: Record<string, string | number | boolean>
}

/** 回传给后端入库的切片（与后端 `rag.models.ChunkIn` 对应）。 */
export interface ChunkIn {
  content: string
  metadata: Record<string, string | number | boolean>
}

/** 切分预览响应（`GET /api/documents/{id}/chunks`）。 */
export interface SplitPreview {
  job_id: string
  file_name: string
  suffix: string
  /** 实际用于切分的 markdown_file（如 full.md）。 */
  source_file: string
  /** 本次切分结果在服务端缓存里的 id：入库时回传即可，无需携带切片正文。 */
  split_id: string
  strategy: string
  strategy_label: string
  chunk_size: number
  chunk_overlap: number
  total: number
  characters: number
  truncated: boolean
  chunks: ChunkOut[]
}

/** 写入 Milvus `rag_chunk` 的结果（`POST /api/documents/{id}/rag`）。 */
export interface RagIngestResult {
  job_id: string
  chunks: number
  characters: number
  database: string
  collection: string
  store: string
  elapsed_seconds: number
  /** cached=按 split_id 读缓存；reviewed=按回传切片；resplit=服务端重新切分。 */
  mode: 'cached' | 'reviewed' | 'resplit'
}

export interface SplitParams {
  chunk_size?: number
  chunk_overlap?: number
  limit?: number
}

/** `POST /api/documents/{id}/rag` 的参数。 */
export interface RagIngestParams {
  /** 切分预览返回的缓存 id（推荐）：服务端按 id 读取完整切片。 */
  split_id?: string
  /** 无状态回传切片内容（多实例 / 脚本）；与 split_id 二选一。 */
  chunks?: ChunkIn[]
  /** 切分总数：与实际入库数量不一致时后端拒绝入库。 */
  total?: number
  chunk_size?: number
  chunk_overlap?: number
}

export interface UploadOptions {
  model_version: string
  language: string
  enable_formula: boolean
  enable_table: boolean
  is_ocr: boolean
  page_ranges?: string
}

/* ---- RAG 记录 / RAG 问答（与后端 `ai_agent_py.rag.models` 对应）---------- */

/** 集合里的一篇文档：块数 + MySQL 任务信息（可选）。 */
export interface RagDocInfo {
  doc_id: string
  chunks: number
  file_name: string | null
  state: string | null
  /** ISO 8601（UTC）。 */
  ingested_at: string | null
}

/** `GET /api/rag/overview`：Milvus `rag_chunk` 概览。 */
export interface RagOverview {
  ready: boolean
  error: string | null
  milvus_uri: string
  database: string
  collection: string
  dim: number
  metric: string
  analyzer: string
  embedding_model: string
  embedding_configured: boolean
  llm_model: string
  llm_configured: boolean
  /** RRF 之后的精排模型。 */
  rerank_model: string
  /** 精排已启用且有 Key；false 表示只用 RRF 排序。 */
  rerank_configured: boolean
  /** 送进精排的候选条数。 */
  rerank_candidates: number
  /** 最终返回给模型的片段数。 */
  retriever_k: number
  total_chunks: number | null
  total_docs: number | null
  docs: RagDocInfo[]
  docs_truncated: boolean
  doc_names_available: boolean
}

/** 一条 `rag_chunk` 记录（不含向量字段）。 */
export interface RagChunkRecord {
  chunk_id: string
  doc_id: string
  doc_name: string | null
  chunk_index: number
  page: number | null
  content: string
  characters: number
}

/** `GET /api/rag/chunks` 的一页记录。 */
export interface RagChunkList {
  total: number
  page: number
  page_size: number
  doc_id: string | null
  keyword: string | null
  truncated: boolean
  items: RagChunkRecord[]
}

export interface RagChunkParams {
  doc_id?: string
  keyword?: string
  page?: number
  page_size?: number
}

/** 答案引用的一条检索片段。 */
export interface RagSource {
  index: number
  chunk_id: string
  doc_id: string
  source: string
  page: number | null
  chunk_index: number
  /** 精排分（0~1）；未精排时是 RRF 融合分。 */
  score: number | null
  /** 这一条是否经过精排重排。 */
  reranked: boolean
  /** 召回方式：hybrid / dense / bm25。 */
  retrieval: string
  snippet: string
}

/** 引用校验结论：ok=通过；weak=支持度弱但保留；invalid=编号越界；unsupported=数字冲突。 */
export type RagCitationStatus = 'ok' | 'weak' | 'invalid' | 'unsupported'

/** 答案里的一条引用及校验结果；被移除的条目 start/end 为 null。 */
export interface RagCitation {
  index: number
  chunk_id: string
  doc_id: string
  quote: string
  start: number | null
  end: number | null
  status: RagCitationStatus
  reason: string
  removed: boolean
}

/** 一次回答的引用统计；total = ok + weak + removed。 */
export interface RagCitationStats {
  total: number
  ok: number
  weak: number
  removed: number
  invalid: number
  unsupported: number
}

/** `POST /api/rag/chat` 的响应。 */
export interface RagChatResponse {
  answer: string
  thread_id: string
  query: string
  sources: RagSource[]
  citations: RagCitation[]
  citation_stats: RagCitationStats
}

/** SSE 事件回调（`POST /api/rag/chat/stream`）。 */
export interface RagChatStreamHandlers {
  onStage?: (stage: string, message: string) => void
  onSources?: (payload: { query: string; sources: RagSource[] }) => void
  onToken?: (text: string) => void
  onDone?: (payload: {
    answer: string
    query: string
    sources: RagSource[]
    citations: RagCitation[]
    citation_stats: RagCitationStats | null
  }) => void
  onError?: (message: string) => void
}

/* ---- Agent 应用（报销审核等） ------------------------------------------ */

/** `GET /api/agents` 的列表项。 */
export interface AgentSummary {
  id: string
  name: string
  summary: string
  status: string
  version: string
  tags: string[]
  node_count: number
  rule_count: number
}

/** LangGraph 导出的节点。 */
export interface AgentGraphNode {
  id: string
  label: string
  gate: string
  kind: string
  description: string
}

export interface AgentGraphEdge {
  source: string
  target: string
  conditional: boolean
}

/** `GET /api/agents/{id}/graph`：mermaid 源码 + 节点/边。 */
export interface AgentGraph {
  mermaid: string
  mermaid_source: string
  nodes: AgentGraphNode[]
  edges: AgentGraphEdge[]
  node_count: number
  edge_count: number
  rule_count: number
  param_status: string
}

export interface AgentDetail extends AgentSummary {
  graph: AgentGraph
}

/** 样例发票（字段与后端 claim.invoices 对齐，供试跑页对照）。 */
export interface ClaimInvoice {
  number: string
  date: string
  seller: string
  buyer: string
  buyer_tax_id: string
  kind: string
  amount: string
  tax: string
  total: string
  items: { name: string; amount: string }[]
  nights: number
  city_tier: string
}

/** 样例报销单原文。 */
export interface ClaimData {
  claim_id: string
  employee: { id?: string; name: string; dept: string }
  period: string
  category: string
  purpose: string
  total: string
  submit_date: string
  ocr_confidence: number
  meta?: { headcount?: number }
  approvals: { role: string; name: string; approved_at?: string }[]
  attachments: string[]
  invoices: ClaimInvoice[]
}

export interface AgentSample {
  sample_id: string
  title: string
  expected: string
  note: string
  /** 单据原文：试跑时与结果逐项对照。 */
  claim: ClaimData
}

export interface AgentFinding {
  rule_id: string
  name: string
  category: string
  severity: string
  stage: string
  message: string
  citation: string
  evidence: Record<string, unknown>
}

export interface AgentTraceEvent {
  node: string
  label: string
  gate: string
  ms: number
  summary: string
}

/** `POST /api/agents/{id}/run` 的结果。 */
export interface AgentRunResult {
  claim_id: string
  status: string
  decision: { status: string; reason: string; confidence: number }
  confidence: number
  findings: AgentFinding[]
  trace: AgentTraceEvent[]
  elapsed_ms: number
}

export interface AgentRule {
  id: string
  name: string
  category: string
  severity: string
  stage: string
  citation: string
  /** 该规则依赖的参数键（对应参数库里的 key）。 */
  params: string[]
}

export interface AgentConfig {
  rules: AgentRule[]
  params: Record<string, unknown>
  param_status: string
  param_note: string
  /** 参数文件路径（相对项目根目录）。 */
  param_file: string
  param_version: string
  /** 最近一次保存时间（ISO 字符串）。 */
  param_updated_at?: string | null
}

/** `POST /api/agents/{id}/upload` 的响应。 */
export interface AgentUploadResult {
  filename: string
  claim: ClaimData
  warnings: string[]
}

/* ---- 通用 Agent（全部工具 + MySQL 会话记忆） ---------------------------- */

/** `GET /api/general-agent/overview` 里的一个工具。 */
export interface GeneralAgentTool {
  name: string
  group: string
  group_label: string
  description: string
  /** 单次调用超时（秒）；null 表示由工具自身控制（shell 的命令超时）。 */
  timeout_seconds?: number | null
  /** 临时失败（网络 / 429 / 5xx）是否自动重试。 */
  retryable?: boolean
  max_retries?: number
  /** low / medium / high。 */
  risk_level?: string
  /** 同一轮是否与其他工具并行执行。 */
  concurrent?: boolean
}

/** MySQL 会话记忆状态（表是否可读写、历史窗口、会话数）。 */
export interface GeneralAgentMemory {
  configured: boolean
  ok: boolean
  tables: string[]
  agent_id: string
  candidate_messages: number
  session_count: number | null
  summary_table_ready: boolean | null
  error: string | null
}

/** 概览页展示的上下文预算配置。 */
export interface GeneralAgentContextConfig {
  model_context_tokens: number
  history_budget_tokens: number
  output_reserve_tokens: number
  token_chars: number
  history_keep_messages: number
  tool_result_max_chars: number
  tool_turn_max_chars: number
  summary_enabled: boolean
  summary_trigger_tokens: number
  summary_keep_messages: number
  summary_max_chars: number
  overflow_retries: number
}

export interface GeneralAgentOverview {
  ready: boolean
  llm_model: string
  llm_configured: boolean
  max_turns: number
  workspace: string
  shell_enabled: boolean
  tools: GeneralAgentTool[]
  memory: GeneralAgentMemory
  context: GeneralAgentContextConfig
  warnings: string[]
}

/** 一轮对话实际使用的上下文统计。 */
export interface AgentContextStats {
  estimated_tokens: number
  history_budget_tokens: number
  model_context_tokens: number
  messages_used: number
  messages_dropped: number
  summary_used: boolean
  summary_chars: number
  compressed_messages: number
  summarized_messages: number
  summary_error: string | null
}

/** 已压缩历史的摘要。 */
export interface AgentChatSummary {
  summary: string
  until_seq: number
  source_messages: number
  tokens: number | null
  model: string | null
  updated_at: string | null
}

/** 一个 MySQL 会话（`GET /api/general-agent/sessions`）。 */
export interface AgentChatSession {
  thread_id: string
  title: string
  message_count: number
  created_at: string
  updated_at: string
}

export type AgentChatRole = 'user' | 'assistant' | 'tool'

/** 会话里的一条消息（user / assistant / 工具调用记录）。 */
export interface AgentChatMessage {
  seq: number
  role: AgentChatRole
  content: string
  /** 是否已被摘要覆盖：仍在 MySQL/界面里，但不再发送给模型。 */
  compressed: boolean
  tool_name: string | null
  tool_call_id: string | null
  tool_args: string | null
  tool_status: string | null
  elapsed_ms: number | null
  created_at: string | null
}

export interface AgentChatMessageList {
  thread_id: string
  total: number
  items: AgentChatMessage[]
  summary: AgentChatSummary | null
}

/** `POST /api/general-agent/chat` 的响应。 */
export interface AgentChatResponse {
  answer: string
  thread_id: string
  session: AgentChatSession
  tool_calls: number
  elapsed_ms: number
  context: AgentContextStats
  summary: AgentChatSummary | null
  messages: AgentChatMessage[]
}

/** 手动压缩结果（`POST /sessions/{id}/compact`）。 */
export interface AgentCompactResult {
  compacted: boolean
  summary: string
  until_seq: number
  source_messages: number
  tokens: number
  summarized: number
  message: string
}

export interface AgentToolCallEvent {
  id: string
  name: string
  args: string | null
  index: number
}

export interface AgentToolResultEvent {
  id: string
  name: string
  status: string
  elapsed_ms: number | null
  content: string
}

/** SSE 事件回调（`POST /api/general-agent/chat/stream`）。 */
export interface AgentChatStreamHandlers {
  onStage?: (stage: string, message: string) => void
  onSession?: (session: AgentChatSession) => void
  onToolCall?: (call: AgentToolCallEvent) => void
  onToolResult?: (result: AgentToolResultEvent) => void
  onToken?: (text: string) => void
  onDone?: (payload: {
    answer: string
    thread_id: string
    session: AgentChatSession
    tool_calls: number
    elapsed_ms: number
    context: AgentContextStats
    summary: AgentChatSummary | null
    messages: AgentChatMessage[]
  }) => void
  onError?: (message: string) => void
}

/* ---- 私域 OSS（与后端 `ai_agent_py.alioss.models` 对应）------------------ */

/** `GET /api/oss/overview`：私有 Bucket 配置与访问要求（不含密钥）。 */
export interface OssOverview {
  configured: boolean
  access_configured: boolean
  access_required: boolean
  bucket: string | null
  endpoint: string | null
  region: string
  prefix: string
  /** OSS 直链有效期（秒）。 */
  url_expires: number
  max_file_mb: number
  accepted_suffixes: string[]
  warnings: string[]
}

/** 上传成功后的图片信息与 OSS 直链。 */
export interface OssImage {
  key: string
  name: string
  size: number
  content_type: string
  etag: string | null
  /** OSS 直链（预签名 GET，可直接放 `<img>`）。 */
  url: string
  /** OSS 直链（Content-Disposition: attachment）。 */
  download_url: string
  /** 上述直链的有效期（秒）。 */
  expires_in: number
  uploaded_at: string
}

/** `GET /api/oss/objects` 里的一个对象。 */
export interface OssObject {
  key: string
  name: string
  size: number
  content_type: string
  last_modified: string | null
  /** OSS 直链（预签名 GET）。 */
  url: string
  /** OSS 直链（附件下载）。 */
  download_url: string
  /** 上述直链的有效期（秒）。 */
  expires_in: number
}

/** `GET /api/oss/presign`：按对象键换取新的 OSS 直链。 */
export interface OssPresign {
  key: string
  url: string
  download_url: string
  expires_in: number
}

export interface OssObjectList {
  prefix: string
  count: number
  truncated: boolean
  next_marker: string | null
  objects: OssObject[]
}

