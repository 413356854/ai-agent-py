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

export interface IngestInfo {
  chunks: number
  characters: number
  collection: string
  store: string
  source: string
  ingested_at: string
}

/** RAG（Milvus `rag_chunk`）入库摘要；与 Chroma 的 `ingest` 分开记录。 */
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
  ingest: IngestInfo | null
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

export interface IngestResult {
  job_id: string
  chunks: number
  characters: number
  collection: string
  store: string
  source: string
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

/** `POST /api/rag/chat` 的响应。 */
export interface RagChatResponse {
  answer: string
  thread_id: string
  query: string
  sources: RagSource[]
}

/** SSE 事件回调（`POST /api/rag/chat/stream`）。 */
export interface RagChatStreamHandlers {
  onStage?: (stage: string, message: string) => void
  onSources?: (payload: { query: string; sources: RagSource[] }) => void
  onToken?: (text: string) => void
  onDone?: (payload: { answer: string; query: string; sources: RagSource[] }) => void
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
