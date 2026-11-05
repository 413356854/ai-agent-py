import type {
  AgentChatMessage,
  AgentChatMessageList,
  AgentChatResponse,
  AgentChatSession,
  AgentChatStreamHandlers,
  AgentChatSummary,
  AgentCompactResult,
  AgentConfig,
  AgentContextStats,
  AgentDetail,
  AgentGraph,
  AgentRunResult,
  AgentSample,
  AgentSummary,
  AgentUploadResult,
  DocumentConfig,
  GeneralAgentOverview,
  MarkdownOut,
  OssImage,
  OssObjectList,
  OssOverview,
  OssPresign,
  ParseJob,
  RagChatResponse,
  RagChatStreamHandlers,
  RagChunkList,
  RagChunkParams,
  RagCitation,
  RagCitationStats,
  RagIngestParams,
  RagIngestResult,
  RagOverview,
  RagSource,
  SplitParams,
  SplitPreview,
  UploadOptions,
} from '../types'

/** 由 FastAPI 直接托管时为相对路径，开发时可被 Vite 代理。 */
export const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch (error) {
    throw new ApiError(0, `无法连接后端服务：${String(error)}`)
  }
  if (!response.ok) {
    let detail = `请求失败（HTTP ${response.status}）`
    try {
      const body: unknown = await response.json()
      if (body && typeof body === 'object' && 'detail' in body) {
        const raw = (body as { detail: unknown }).detail
        detail = typeof raw === 'string' ? raw : JSON.stringify(raw)
      }
    } catch {
      // 响应不是 JSON，保留默认提示
    }
    throw new ApiError(response.status, detail)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

function encodeOssKey(key: string): string {
  return key
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/')
}

export const api = {
  config: () => request<DocumentConfig>('/api/documents/config'),

  list: () => request<ParseJob[]>('/api/documents'),

  get: (jobId: string) => request<ParseJob>(`/api/documents/${jobId}`),

  parse: (file: File, options: UploadOptions) => {
    const form = new FormData()
    form.append('file', file)
    form.append('model_version', options.model_version)
    form.append('language', options.language)
    form.append('enable_formula', String(options.enable_formula))
    form.append('enable_table', String(options.enable_table))
    form.append('is_ocr', String(options.is_ocr))
    if (options.page_ranges?.trim()) {
      form.append('page_ranges', options.page_ranges.trim())
    }
    return request<ParseJob>('/api/documents/parse', { method: 'POST', body: form })
  },

  markdown: (jobId: string) => request<MarkdownOut>(`/api/documents/${jobId}/markdown`),

  /** 按文档后缀选择切分器，返回预览片段。 */
  split: (jobId: string, params: SplitParams = {}) => {
    const query = new URLSearchParams()
    if (params.chunk_size !== undefined) query.set('chunk_size', String(params.chunk_size))
    if (params.chunk_overlap !== undefined) {
      query.set('chunk_overlap', String(params.chunk_overlap))
    }
    if (params.limit !== undefined) query.set('limit', String(params.limit))
    const suffix = query.size ? `?${query.toString()}` : ''
    return request<SplitPreview>(`/api/documents/${jobId}/chunks${suffix}`)
  },

  /**
   * 把用户查阅过的切分结果写回 Milvus `rag_chunk`（按 doc_id 覆盖旧块）。
   * 优先传 split_id：后端从缓存读取完整切片，请求体只有几十字节。
   */
  ingestRag: (jobId: string, params: RagIngestParams = {}) =>
    request<RagIngestResult>(`/api/documents/${jobId}/rag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...(params.split_id !== undefined ? { split_id: params.split_id } : {}),
        ...(params.chunks !== undefined ? { chunks: params.chunks } : {}),
        ...(params.total !== undefined ? { total: params.total } : {}),
        ...(params.chunk_size !== undefined ? { chunk_size: params.chunk_size } : {}),
        ...(params.chunk_overlap !== undefined
          ? { chunk_overlap: params.chunk_overlap }
          : {}),
      }),
    }),

  remove: (jobId: string) => request<void>(`/api/documents/${jobId}`, { method: 'DELETE' }),

  downloadUrl: (jobId: string, kind: 'markdown' | 'zip' | 'source') =>
    `${API_BASE}/api/documents/${jobId}/download?kind=${kind}`,

  /* ---- RAG 记录 / RAG 问答 -------------------------------------------- */
  ragOverview: () => request<RagOverview>('/api/rag/overview'),

  ragChunks: (params: RagChunkParams = {}) => {
    const query = new URLSearchParams()
    if (params.doc_id) query.set('doc_id', params.doc_id)
    if (params.keyword) query.set('keyword', params.keyword)
    if (params.page !== undefined) query.set('page', String(params.page))
    if (params.page_size !== undefined) query.set('page_size', String(params.page_size))
    const suffix = query.size ? `?${query.toString()}` : ''
    return request<RagChunkList>(`/api/rag/chunks${suffix}`)
  },

  /** 非流式问答；聊天界面用 `ragChatStream` 逐 token 显示。`docIds` 非空时只在这些文档内检索。 */
  ragChat: (message: string, threadId: string, docIds: string[] = []) =>
    request<RagChatResponse>('/api/rag/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, thread_id: threadId, doc_ids: docIds }),
    }),

  /** 流式问答：`stage` → `sources` → 若干 `token` → `done`（异常走 `error`）。 */
  ragChatStream: (
    message: string,
    threadId: string,
    docIds: string[],
    handlers: RagChatStreamHandlers,
    signal?: AbortSignal,
  ) => streamRagChat(message, threadId, docIds, handlers, signal),

  /* ---- Agent 应用（报销审核等）--------------------------------------- */
  agents: () => request<AgentSummary[]>('/api/agents'),

  agent: (agentId: string) => request<AgentDetail>(`/api/agents/${agentId}`),

  agentGraph: (agentId: string) => request<AgentGraph>(`/api/agents/${agentId}/graph`),

  agentSamples: (agentId: string) => request<AgentSample[]>(`/api/agents/${agentId}/samples`),

  agentConfig: (agentId: string) => request<AgentConfig>(`/api/agents/${agentId}/config`),

  /** 整体覆盖写入参数并刷新缓存（落盘成功后返回最新配置）。 */
  updateAgentParams: (agentId: string, params: Record<string, unknown>) =>
    request<AgentConfig>(`/api/agents/${agentId}/config/params`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params }),
    }),

  /** 手工改了 YAML 后，让进程重新读文件。 */
  reloadAgentConfig: (agentId: string) =>
    request<AgentConfig>(`/api/agents/${agentId}/config/reload`, { method: 'POST' }),

  runAgent: (agentId: string, sampleId: string) =>
    request<AgentRunResult>(`/api/agents/${agentId}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id: sampleId }),
    }),

  /** 用上传解析出来的单据试跑。 */
  runAgentClaim: (agentId: string, claim: Record<string, unknown>) =>
    request<AgentRunResult>(`/api/agents/${agentId}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ claim }),
    }),

  /** 上传单据（.xlsx / .json），返回解析后的 claim 与告警。 */
  parseUpload: (agentId: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<AgentUploadResult>(`/api/agents/${agentId}/upload`, {
      method: 'POST',
      body: form,
    })
  },

  /** 空白模板 xlsx 下载地址。 */
  agentTemplateUrl: (agentId: string) => `${API_BASE}/api/agents/${agentId}/template`,

  /** 样例单据下载地址（填充好的样板）。 */
  agentSampleDownloadUrl: (agentId: string, sampleId: string, fmt: 'xlsx' | 'json' = 'xlsx') =>
    `${API_BASE}/api/agents/${agentId}/samples/${sampleId}/download?fmt=${fmt}`,

  /* ---- 通用 Agent（全部工具 + MySQL 会话记忆）------------------------- */
  generalAgentOverview: () => request<GeneralAgentOverview>('/api/general-agent/overview'),

  generalSessions: (limit = 50) =>
    request<AgentChatSession[]>(`/api/general-agent/sessions?limit=${limit}`),

  generalSessionMessages: (threadId: string, limit = 500) =>
    request<AgentChatMessageList>(
      `/api/general-agent/sessions/${encodeURIComponent(threadId)}/messages?limit=${limit}`,
    ),

  deleteGeneralSession: (threadId: string) =>
    request<void>(`/api/general-agent/sessions/${encodeURIComponent(threadId)}`, {
      method: 'DELETE',
    }),

  /** 手动压缩更早历史（不删除原始消息，只更新摘要）。 */
  compactGeneralSession: (threadId: string, keep?: number) =>
    request<AgentCompactResult>(
      `/api/general-agent/sessions/${encodeURIComponent(threadId)}/compact`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(keep === undefined ? {} : { keep }),
      },
    ),

  /** 非流式对话；页面用 `generalChatStream` 逐 token 显示并实时展示工具调用。 */
  generalChat: (message: string, threadId: string, images: string[] = []) =>
    request<AgentChatResponse>('/api/general-agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, thread_id: threadId, images }),
    }),

  generalChatStream: (
    message: string,
    threadId: string,
    handlers: AgentChatStreamHandlers,
    signal?: AbortSignal,
    images: string[] = [],
  ) => streamGeneralChat(message, threadId, handlers, signal, images),

  /* ---- 私域 OSS（阿里云私有 Bucket：上传 / 读取 / 列举 / 删除）-------- */
  ossOverview: () => request<OssOverview>('/api/oss/overview'),

  /** 上传图片；返回 OSS 直链（预览 + 附件下载）。 */
  ossUpload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<OssImage>('/api/oss/images', {
      method: 'POST',
      body: form,
    })
  },

  /** `OSS_PREFIX/` 下的对象，最新在前；每项都带 OSS 直链。 */
  ossObjects: (limit = 50) => request<OssObjectList>(`/api/oss/objects?limit=${limit}`),

  /** 按对象键换取新的 OSS 直链（预览 + 下载）。 */
  ossPresign: (key: string) => {
    const query = new URLSearchParams({ key })
    return request<OssPresign>(`/api/oss/presign?${query.toString()}`)
  },

  ossDelete: (key: string) =>
    request<void>(`/api/oss/images/${encodeOssKey(key)}`, { method: 'DELETE' }),
}

/** Markdown 里的相对图片路径（images/xxx.jpg）→ 解析结果静态目录。 */
export function resolveAsset(base: string, source: string): string {
  if (/^(https?:)?\/\//i.test(source) || source.startsWith('/') || source.startsWith('data:')) {
    return source
  }
  return `${API_BASE}${base.replace(/\/$/, '')}/${source.replace(/^\.\//, '')}`
}

/* ---- SSE：RAG 问答流式输出 -----------------------------------------------
 * 后端用 `text/event-stream` 推送四类事件（event: 名称 + data: JSON）：
 *   stage   {stage, message}                 检索阶段提示
 *   sources {query, sources[]}               引用片段（在 token 之前）
 *   token   {text}                           增量答案
 *   done    {answer, query, sources[]}       完整答案（以它为准）
 *   error   {message}                        失败原因
 * 这里手写一个极简 SSE 解析器：fetch + ReadableStream，支持 AbortSignal。
 */
interface SseEvent {
  event: string
  data: string
}

function parseSseBlock(raw: string): SseEvent | null {
  let event = 'message'
  const data: string[] = []
  for (const line of raw.split(/\r?\n/)) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim()
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).trimStart())
  }
  if (!data.length) return null
  return { event, data: data.join('\n') }
}

async function* readSseEvents(response: Response): AsyncGenerator<SseEvent> {
  const reader = response.body?.getReader()
  if (!reader) throw new ApiError(0, '当前浏览器不支持流式响应')
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const drain = (): SseEvent[] => {
    const events: SseEvent[] = []
    let match = /\r?\n\r?\n/.exec(buffer)
    while (match) {
      const parsed = parseSseBlock(buffer.slice(0, match.index))
      buffer = buffer.slice(match.index + match[0].length)
      if (parsed) events.push(parsed)
      match = /\r?\n\r?\n/.exec(buffer)
    }
    return events
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    for (const event of drain()) yield event
  }
  buffer += decoder.decode()
  for (const event of drain()) yield event
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

/** 打开一个 POST + SSE 的流，返回事件迭代器；HTTP 错误按 `ApiError` 抛出。 */
async function openEventStream(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<AsyncGenerator<SseEvent>> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
      signal,
    })
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new ApiError(0, `无法连接后端服务：${String(error)}`)
  }

  if (!response.ok) {
    let detail = `请求失败（HTTP ${response.status}）`
    try {
      const payload: unknown = await response.json()
      if (payload && typeof payload === 'object' && 'detail' in payload) {
        const raw = (payload as { detail: unknown }).detail
        detail = typeof raw === 'string' ? raw : JSON.stringify(raw)
      }
    } catch {
      // 响应不是 JSON：保留默认提示
    }
    throw new ApiError(response.status, detail)
  }

  return readSseEvents(response)
}

async function streamRagChat(
  message: string,
  threadId: string,
  docIds: string[],
  handlers: RagChatStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const events = await openEventStream(
    '/api/rag/chat/stream',
    { message, thread_id: threadId, doc_ids: docIds },
    signal,
  )

  for await (const { event, data } of events) {
    let payload: Record<string, unknown>
    try {
      payload = JSON.parse(data) as Record<string, unknown>
    } catch {
      continue
    }
    const sources = (payload.sources as RagSource[] | undefined) ?? []
    const citations = (payload.citations as RagCitation[] | undefined) ?? []
    const citationStats = (payload.citation_stats as RagCitationStats | undefined) ?? null
    if (event === 'stage') {
      handlers.onStage?.(String(payload.stage ?? ''), String(payload.message ?? ''))
    } else if (event === 'sources') {
      handlers.onSources?.({ query: String(payload.query ?? ''), sources })
    } else if (event === 'token') {
      handlers.onToken?.(String(payload.text ?? ''))
    } else if (event === 'done') {
      handlers.onDone?.({
        answer: String(payload.answer ?? ''),
        query: String(payload.query ?? ''),
        sources,
        citations,
        citation_stats: citationStats,
      })
    } else if (event === 'error') {
      handlers.onError?.(String(payload.message ?? '问答失败'))
      return
    }
  }
}

function asSession(value: unknown): AgentChatSession {
  const raw = (value ?? {}) as Record<string, unknown>
  return {
    thread_id: String(raw.thread_id ?? ''),
    title: String(raw.title ?? ''),
    message_count: Number(raw.message_count ?? 0),
    created_at: String(raw.created_at ?? ''),
    updated_at: String(raw.updated_at ?? ''),
  }
}

async function streamGeneralChat(
  message: string,
  threadId: string,
  handlers: AgentChatStreamHandlers,
  signal?: AbortSignal,
  images: string[] = [],
): Promise<void> {
  const events = await openEventStream(
    '/api/general-agent/chat/stream',
    { message, thread_id: threadId, images },
    signal,
  )

  for await (const { event, data } of events) {
    let payload: Record<string, unknown>
    try {
      payload = JSON.parse(data) as Record<string, unknown>
    } catch {
      continue
    }
    if (event === 'stage') {
      handlers.onStage?.(String(payload.stage ?? ''), String(payload.message ?? ''))
    } else if (event === 'session') {
      handlers.onSession?.(asSession(payload))
    } else if (event === 'tool_call') {
      handlers.onToolCall?.({
        id: String(payload.id ?? ''),
        name: String(payload.name ?? ''),
        args: payload.args == null ? null : String(payload.args),
        index: Number(payload.index ?? 0),
      })
    } else if (event === 'tool_result') {
      handlers.onToolResult?.({
        id: String(payload.id ?? ''),
        name: String(payload.name ?? ''),
        status: String(payload.status ?? 'ok'),
        elapsed_ms: payload.elapsed_ms == null ? null : Number(payload.elapsed_ms),
        content: String(payload.content ?? ''),
      })
    } else if (event === 'token') {
      handlers.onToken?.(String(payload.text ?? ''))
    } else if (event === 'done') {
      handlers.onDone?.({
        answer: String(payload.answer ?? ''),
        thread_id: String(payload.thread_id ?? ''),
        session: asSession(payload.session),
        tool_calls: Number(payload.tool_calls ?? 0),
        elapsed_ms: Number(payload.elapsed_ms ?? 0),
        context: (payload.context ?? {}) as AgentContextStats,
        summary: (payload.summary ?? null) as AgentChatSummary | null,
        messages: Array.isArray(payload.messages)
          ? (payload.messages as AgentChatMessage[])
          : [],
      })
    } else if (event === 'error') {
      handlers.onError?.(String(payload.message ?? '通用 Agent 执行失败'))
      return
    }
  }
}
