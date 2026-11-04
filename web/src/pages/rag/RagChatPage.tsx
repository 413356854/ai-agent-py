import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import {
  BulbOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  ReloadOutlined,
  RobotOutlined,
  SendOutlined,
  StopOutlined,
  UserOutlined,
} from '@ant-design/icons'
import {
  Alert,
  Button,
  Empty,
  Input,
  Popover,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
} from 'antd'
import { useNavigate } from 'react-router-dom'
import { ApiError, api } from '../../api/client'
import { MarkdownPreview } from '../../components/MarkdownPreview'
import type { RagOverview, RagSource } from '../../types'

type MessageStatus = 'loading' | 'streaming' | 'done' | 'error' | 'stopped'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  status: MessageStatus
  sources: RagSource[]
  query?: string
  stage?: string
  error?: string
}

const EXAMPLES = [
  '企业所得税的扣除项目有哪些？',
  '业务招待费的扣除标准是什么？',
  '哪些支出不得在企业所得税前扣除？',
]

function newId(): string {
  const cryptoApi = globalThis.crypto
  if (cryptoApi && typeof cryptoApi.randomUUID === 'function') return cryptoApi.randomUUID()
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

function errorText(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return String(error)
}

function sourceLabel(source: RagSource): string {
  return source.source.length > 26 ? `${source.source.slice(0, 26)}…` : source.source
}

interface MessageItemProps {
  message: ChatMessage
  onOpenRecords: (docId: string) => void
}

/** 单条消息：流式回答只更新最后一条，memo 避免历史消息反复重渲染 Markdown。 */
const MessageItem = memo(function MessageItem({ message, onOpenRecords }: MessageItemProps) {
  if (message.role === 'user') {
    return (
      <div className="chat-msg chat-msg-user">
        <div className="chat-avatar">
          <UserOutlined />
        </div>
        <div className="chat-body">
          <div className="chat-bubble">
            <div className="chat-plain">{message.content}</div>
          </div>
        </div>
      </div>
    )
  }

  const waiting = message.status === 'loading' || message.status === 'streaming'
  const waitLabel = message.sources.length ? '正在生成回答…' : (message.stage ?? '正在检索知识库…')
  const docId = message.sources[0]?.doc_id

  return (
    <div className="chat-msg chat-msg-assistant">
      <div className="chat-avatar">
        <RobotOutlined />
      </div>
      <div className="chat-body">
        <div className="chat-bubble">
          {message.content ? <MarkdownPreview markdown={message.content} assetsBase="" /> : null}
          {waiting && (
            <div className="chat-loading">
              <Spin size="small" />
              <span>{waitLabel}</span>
            </div>
          )}
          {message.status === 'streaming' && message.content && <span className="chat-cursor" />}
          {message.status === 'error' && (
            <Alert
              type="error"
              showIcon
              style={{ marginTop: message.content ? 8 : 0 }}
              message="回答失败"
              description={message.error ?? '请稍后重试'}
            />
          )}
          {message.status === 'stopped' && !message.content && (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="已停止生成" />
          )}
        </div>

        {message.status === 'stopped' && message.content && (
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            已停止生成
          </Typography.Text>
        )}

        {message.sources.length > 0 && (
          <div className="chat-sources">
            <Space size={8} wrap>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                引用片段（{message.sources.length}）
                {message.sources.some((source) => source.reranked) ? ' · 已精排' : ''}
              </Typography.Text>
              {message.query && (
                <Typography.Text type="secondary" style={{ fontSize: 12 }} ellipsis>
                  检索：{message.query}
                </Typography.Text>
              )}
            </Space>
            <Space size={[6, 6]} wrap>
              {message.sources.map((source) => (
                <Popover
                  key={`${source.chunk_id}-${source.index}`}
                  placement="top"
                  title={
                    <span className="chat-source-title">
                      [{source.index}] {source.source}
                      {source.page ? ` · 第 ${source.page} 页` : ''} · 块 #
                      {source.chunk_index + 1}
                      {source.score != null ? ` · 相关度 ${source.score.toFixed(3)}` : ''}
                    </span>
                  }
                  content={<div className="chat-source-pop">{source.snippet}</div>}
                >
                  <Tag className="chat-source-tag" icon={<DatabaseOutlined />}>
                    [{source.index}] {sourceLabel(source)}
                    {source.page ? ` · p.${source.page}` : ''}
                  </Tag>
                </Popover>
              ))}
              {docId && (
                <Button type="link" size="small" onClick={() => onOpenRecords(docId)}>
                  在 RAG 记录中查看
                </Button>
              )}
            </Space>
          </div>
        )}
      </div>
    </div>
  )
})

/** RAG 问答：LangGraph 多轮检索问答 + SSE 流式输出。 */
export function RagChatPage() {
  const navigate = useNavigate()
  const [threadId, setThreadId] = useState(newId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [overview, setOverview] = useState<RagOverview | null>(null)
  const [overviewError, setOverviewError] = useState('')
  const [loadingStatus, setLoadingStatus] = useState(true)

  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const loadOverview = useCallback(async () => {
    setLoadingStatus(true)
    try {
      const next = await api.ragOverview()
      setOverview(next)
      setOverviewError('')
    } catch (error) {
      setOverviewError(errorText(error))
    } finally {
      setLoadingStatus(false)
    }
  }, [])

  useEffect(() => {
    void loadOverview()
  }, [loadOverview])

  useEffect(() => {
    const element = scrollRef.current
    if (!element) return
    const nearBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 180
    if (nearBottom) element.scrollTop = element.scrollHeight
  }, [messages])

  const canChat = Boolean(
    overview?.ready && overview.llm_configured && (overview.total_chunks ?? 0) > 0,
  )
  const retrievalLabel = overview?.embedding_configured ? '混合检索（向量 + BM25）' : 'BM25 检索'

  const examples = useMemo(() => {
    const fileName = overview?.docs?.[0]?.file_name
    if (!fileName) return EXAMPLES
    const title = fileName.replace(/\.[^.]+$/, '')
    return [`《${title}》主要规定了哪些内容？`, EXAMPLES[1], EXAMPLES[2]]
  }, [overview])

  const openRecords = useCallback(
    (docId: string) => {
      navigate(`/rag/records?doc_id=${encodeURIComponent(docId)}`)
    },
    [navigate],
  )

  const patchMessage = useCallback((id: string, updater: (message: ChatMessage) => ChatMessage) => {
    setMessages((previous) =>
      previous.map((message) => (message.id === id ? updater(message) : message)),
    )
  }, [])

  const send = useCallback(
    async (text: string) => {
      const content = text.trim()
      if (!content || streaming) return

      const assistantId = newId()
      setMessages((previous) => [
        ...previous,
        { id: newId(), role: 'user', content, status: 'done', sources: [] },
        { id: assistantId, role: 'assistant', content: '', status: 'loading', sources: [] },
      ])
      setInput('')
      setStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        await api.ragChatStream(
          content,
          threadId,
          {
            onStage: (stage, stageMessage) =>
              patchMessage(assistantId, (current) => ({
                ...current,
                stage: stageMessage || stage || current.stage,
                status: current.content ? 'streaming' : 'loading',
              })),
            onSources: ({ query, sources }) =>
              patchMessage(assistantId, (current) => ({
                ...current,
                query,
                sources,
                status: current.content ? 'streaming' : 'loading',
              })),
            onToken: (token) =>
              patchMessage(assistantId, (current) => ({
                ...current,
                content: current.content + token,
                status: 'streaming',
              })),
            onDone: ({ answer, query, sources }) =>
              patchMessage(assistantId, (current) => ({
                ...current,
                content: answer || current.content,
                query: query || current.query,
                sources: sources.length ? sources : current.sources,
                status: 'done',
              })),
            onError: (detail) =>
              patchMessage(assistantId, (current) => ({
                ...current,
                status: 'error',
                error: detail,
              })),
          },
          controller.signal,
        )
        if (controller.signal.aborted) {
          patchMessage(assistantId, (current) => ({ ...current, status: 'stopped' }))
        } else {
          patchMessage(assistantId, (current) =>
            current.status === 'loading' || current.status === 'streaming'
              ? { ...current, status: 'done' }
              : current,
          )
        }
      } catch (error) {
        if (controller.signal.aborted) {
          patchMessage(assistantId, (current) => ({ ...current, status: 'stopped' }))
        } else {
          patchMessage(assistantId, (current) => ({
            ...current,
            status: 'error',
            error: errorText(error),
          }))
        }
      } finally {
        setStreaming(false)
        abortRef.current = null
      }
    },
    [patchMessage, streaming, threadId],
  )

  const stop = () => {
    abortRef.current?.abort()
  }

  const reset = () => {
    abortRef.current?.abort()
    setMessages([])
    setInput('')
    setThreadId(newId())
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey) return
    if (event.nativeEvent.isComposing) return
    event.preventDefault()
    if (!streaming && canChat) void send(input)
  }

  const handleReload = () => {
    void loadOverview()
  }

  const showWelcome = messages.length === 0

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <Typography.Title level={4} style={{ margin: 0 }}>
            RAG 问答
          </Typography.Title>
          <Typography.Text type="secondary">
            LangGraph 编排：追问改写 → Milvus 检索
            {overview?.rerank_configured ? ' + 精排' : ''} → 带引用回答
            {overview ? ` · ${overview.database}.${overview.collection}` : ''}
          </Typography.Text>
        </div>
        <Space wrap>
          {overview && (
            <>
              <Tag color="geekblue">{overview.llm_model}</Tag>
              <Tag color={overview.embedding_configured ? 'green' : 'orange'}>
                {retrievalLabel}
              </Tag>
              <Tag color={overview.rerank_configured ? 'purple' : 'orange'}>
                {overview.rerank_configured
                  ? `精排 ${overview.rerank_model}`
                  : '未启用精排'}
              </Tag>
            </>
          )}
          <Tooltip title="重新读取集合与模型配置">
            <Button icon={<ReloadOutlined />} loading={loadingStatus} onClick={handleReload}>
              刷新状态
            </Button>
          </Tooltip>
          <Button icon={<DeleteOutlined />} onClick={reset} disabled={!messages.length && !streaming}>
            新会话
          </Button>
        </Space>
      </div>

      {overviewError && (
        <Alert
          type="error"
          showIcon
          message="无法读取 RAG 状态"
          description={`${overviewError} 请确认后端服务与 Milvus 正常后点「刷新状态」。`}
        />
      )}
      {overview?.ready && !overview.llm_configured && (
        <Alert
          type="warning"
          showIcon
          message="尚未配置对话模型 Key，问答不可用"
          description="在 .env 里设置 LLM_API_KEY，或直接导出环境变量 DASHSCOPE_API_KEY（任何 OpenAI 兼容端点）后重启服务。"
        />
      )}
      {overview?.ready && (overview.total_chunks ?? 0) === 0 && (
        <Alert
          type="warning"
          showIcon
          message="知识库还没有切片"
          description="先到「文档解析」上传文档，切分后点「入库 RAG」，再回来提问。"
        />
      )}
      {overview?.ready && !overview.embedding_configured && (overview.total_chunks ?? 0) > 0 && (
        <Alert
          type="info"
          showIcon
          message="未配置 Embedding Key，本次只使用 BM25 关键词检索"
          description="配置 EMBEDDING_API_KEY（或 DASHSCOPE_API_KEY）后即可启用向量 + 关键词混合检索。"
        />
      )}
      {overview?.ready && !overview.rerank_configured && (overview.total_chunks ?? 0) > 0 && (
        <Alert
          type="info"
          showIcon
          message="未启用精排：当前只按 RRF 融合顺序取前几条"
          description={`配置 RERANK_API_KEY（或 DASHSCOPE_API_KEY）后，前 ${overview.rerank_candidates} 条 RRF 候选会再交给 ${overview.rerank_model} 重排，最终取前 ${overview.retriever_k} 条。`}
        />
      )}

      <div className="chat-panel panel">
        <div className="chat-scroll" ref={scrollRef}>
          {showWelcome ? (
            <div className="chat-welcome">
              <div className="chat-welcome-icon">
                <RobotOutlined />
              </div>
              <Typography.Title level={4}>向知识库提问</Typography.Title>
              <Typography.Paragraph type="secondary" style={{ maxWidth: 640 }}>
                回答只基于已入库的切片；点击引用编号可以查看命中的原文片段。多轮追问会结合上文自动改写检索问题。
              </Typography.Paragraph>
              <Space wrap className="chat-examples">
                {examples.map((example) => (
                  <Button
                    key={example}
                    icon={<BulbOutlined />}
                    disabled={!canChat}
                    onClick={() => void send(example)}
                  >
                    {example}
                  </Button>
                ))}
              </Space>
            </div>
          ) : (
            messages.map((message) => (
              <MessageItem key={message.id} message={message} onOpenRecords={openRecords} />
            ))
          )}
        </div>

        <div className="chat-composer">
          <Input.TextArea
            autoSize={{ minRows: 2, maxRows: 6 }}
            value={input}
            disabled={!canChat}
            placeholder={
              canChat
                ? '输入问题，Enter 发送，Shift + Enter 换行'
                : '知识库 / 模型未就绪，请先完成配置'
            }
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
          />
          {streaming ? (
            <Button danger icon={<StopOutlined />} onClick={stop}>
              停止
            </Button>
          ) : (
            <Button
              type="primary"
              icon={<SendOutlined />}
              disabled={!canChat || !input.trim()}
              onClick={() => void send(input)}
            >
              发送
            </Button>
          )}
        </div>
        <div className="chat-composer-hint">
          会话 ID {threadId.slice(0, 8)} · 上下文保存在服务端进程内，重启服务或点「新会话」后重新开始
        </div>
      </div>
    </div>
  )
}
