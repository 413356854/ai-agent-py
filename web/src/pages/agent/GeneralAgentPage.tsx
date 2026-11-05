import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import {
  BulbOutlined,
  CompressOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  ReloadOutlined,
  RobotOutlined,
  SendOutlined,
  StopOutlined,
  ThunderboltOutlined,
  ToolOutlined,
  UserOutlined,
} from '@ant-design/icons'
import {
  Alert,
  App,
  Button,
  Collapse,
  Drawer,
  Empty,
  Input,
  Popconfirm,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
} from 'antd'
import dayjs from 'dayjs'
import { ApiError, api } from '../../api/client'
import { MarkdownPreview } from '../../components/MarkdownPreview'
import type {
  AgentChatMessage,
  AgentChatSession,
  AgentChatStreamHandlers,
  AgentChatSummary,
  AgentContextStats,
  GeneralAgentOverview,
  GeneralAgentTool,
} from '../../types'

/** 欢迎页示例：分别演示文件工具、知识库检索、联网搜索。 */
const EXAMPLES = [
  '用 glob 找出项目里的 Python 文件，再挑一个用 read 看看它做什么',
  '在知识库里查一下「业务招待费的扣除标准」，并给出引用片段',
  '联网搜索 LangGraph 1.x 的 checkpointer 用法，并总结要点',
]

const TOOL_GROUP_ORDER = ['knowledge', 'filesystem', 'shell', 'web', 'other']

type ToolStatus = 'running' | 'ok' | 'error'

interface ToolTrace {
  id: string
  name: string
  args: string | null
  status: ToolStatus
  result: string
  elapsedMs: number | null
}

type AssistantStatus = 'loading' | 'streaming' | 'done' | 'stopped' | 'error'

interface UserItem {
  kind: 'user'
  id: string
  seq?: number
  content: string
  compressed?: boolean
}

interface AssistantItem {
  kind: 'assistant'
  id: string
  seq?: number
  content: string
  status: AssistantStatus
  tools: ToolTrace[]
  stage?: string
  compressed?: boolean
  error?: string
}

interface ToolItem {
  kind: 'tool'
  id: string
  seq?: number
  tool: ToolTrace
  compressed?: boolean
}

type ChatItem = UserItem | AssistantItem | ToolItem

function newId(): string {
  const cryptoApi = globalThis.crypto
  if (cryptoApi && typeof cryptoApi.randomUUID === 'function') return cryptoApi.randomUUID()
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

function errorText(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return String(error)
}

function formatTime(value: string): string {
  const parsed = dayjs(value)
  return parsed.isValid() ? parsed.format('MM-DD HH:mm') : ''
}

function formatTokens(value: number | null | undefined): string {
  if (value == null || value <= 0) return '—'
  return value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(value)
}

/** MySQL 里的消息 → 页面渲染项（工具调用记录单独成卡片）。 */
function toItems(messages: AgentChatMessage[], includeCompressed = false): ChatItem[] {
  return messages
    .filter((message) => includeCompressed || !message.compressed)
    .map((message, index) => {
      const key = message.seq || index + 1
      if (message.role === 'user') {
        return {
          kind: 'user',
          id: `user-${key}`,
          seq: message.seq,
          content: message.content,
          compressed: message.compressed,
        }
      }
      if (message.role === 'tool') {
        return {
          kind: 'tool',
          id: `tool-${key}`,
          seq: message.seq,
          compressed: message.compressed,
          tool: {
            id: message.tool_call_id ?? `tool-${key}`,
            name: message.tool_name ?? 'tool',
            args: message.tool_args,
            status: message.tool_status === 'error' ? 'error' : 'ok',
            result: message.content,
            elapsedMs: message.elapsed_ms,
          },
        }
      }
      return {
        kind: 'assistant',
        id: `assistant-${key}`,
        seq: message.seq,
        content: message.content,
        status: 'done',
        tools: [],
        compressed: message.compressed,
      }
    })
}

/** 一次工具调用的卡片：名称、入参、状态、耗时可展开的结果。 */
const ToolCard = memo(function ToolCard({ tool }: { tool: ToolTrace }) {
  const status =
    tool.status === 'running'
      ? { color: 'processing', label: '执行中' }
      : tool.status === 'error'
        ? { color: 'red', label: '失败' }
        : { color: 'green', label: '完成' }

  return (
    <div className={`agent-tool-card${tool.status === 'error' ? ' agent-tool-card-error' : ''}`}>
      <Space size={6} wrap>
        <Tag color="geekblue" icon={<ToolOutlined />}>
          {tool.name}
        </Tag>
        <Tag color={status.color}>{status.label}</Tag>
        {tool.elapsedMs != null && (
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {tool.elapsedMs} ms
          </Typography.Text>
        )}
      </Space>
      {tool.args && (
        <div className="agent-tool-args">
          <code>{tool.args}</code>
        </div>
      )}
      {tool.result ? (
        <Collapse
          size="small"
          ghost
          items={[
            {
              key: 'result',
              label: `工具结果（${tool.result.length} 字）`,
              children: <pre className="agent-tool-result">{tool.result}</pre>,
            },
          ]}
        />
      ) : tool.status === 'running' ? (
        <div className="chat-loading">
          <Spin size="small" />
          <span>正在执行…</span>
        </div>
      ) : null}
    </div>
  )
})

const MessageItem = memo(function MessageItem({ item }: { item: ChatItem }) {
  if (item.kind === 'user') {
    return (
      <div className="chat-msg chat-msg-user">
        <div className="chat-avatar">
          <UserOutlined />
        </div>
        <div className="chat-body">
          {item.compressed && (
            <Tag className="agent-compressed-tag">已压缩 · 不再发送给模型</Tag>
          )}
          <div className="chat-bubble">
            <div className="chat-plain">{item.content}</div>
          </div>
        </div>
      </div>
    )
  }

  if (item.kind === 'tool') {
    return (
      <div className="chat-msg chat-msg-tool">
        <div className="chat-avatar">
          <ToolOutlined />
        </div>
        <div className="chat-body">
          {item.compressed && (
            <Tag className="agent-compressed-tag">已压缩 · 不再发送给模型</Tag>
          )}
          <ToolCard tool={item.tool} />
        </div>
      </div>
    )
  }

  const waiting = item.status === 'loading' || item.status === 'streaming'
  const waitLabel = item.status === 'loading' ? '正在调用工具…' : '正在生成回答…'

  return (
    <div className="chat-msg chat-msg-assistant">
      <div className="chat-avatar">
        <RobotOutlined />
      </div>
      <div className="chat-body">
        {item.compressed && (
          <Tag className="agent-compressed-tag">已压缩 · 不再发送给模型</Tag>
        )}
        {item.tools.length > 0 && (
          <div className="agent-tool-list">
            {item.tools.map((tool) => (
              <ToolCard key={tool.id} tool={tool} />
            ))}
          </div>
        )}
        <div className="chat-bubble">
          {item.content ? <MarkdownPreview markdown={item.content} assetsBase="" /> : null}
          {waiting && (
            <div className="chat-loading">
              <Spin size="small" />
              <span>{item.stage || waitLabel}</span>
            </div>
          )}
          {item.status === 'streaming' && item.content && <span className="chat-cursor" />}
          {item.status === 'error' && (
            <Alert
              type="error"
              showIcon
              style={{ marginTop: item.content ? 8 : 0 }}
              message="执行失败"
              description={item.error ?? '请稍后重试'}
            />
          )}
          {item.status === 'stopped' && !item.content && (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="已停止生成" />
          )}
        </div>
        {item.status === 'stopped' && item.content && (
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            已停止生成
          </Typography.Text>
        )}
      </div>
    </div>
  )
})

/** 通用 Agent：全部工具 + MySQL 会话记忆 + SSE 流式（工具调用实时展示）。 */
export function GeneralAgentPage() {
  const { message: toast } = App.useApp()
  const [overview, setOverview] = useState<GeneralAgentOverview | null>(null)
  const [loadingOverview, setLoadingOverview] = useState(true)
  const [overviewError, setOverviewError] = useState('')
  const [sessions, setSessions] = useState<AgentChatSession[]>([])
  const [loadingSessions, setLoadingSessions] = useState(false)
  const [sessionError, setSessionError] = useState('')
  const [threadId, setThreadId] = useState('')
  const [items, setItems] = useState<ChatItem[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [toolsOpen, setToolsOpen] = useState(false)
  const [summaryInfo, setSummaryInfo] = useState<AgentChatSummary | null>(null)
  const [lastContext, setLastContext] = useState<AgentContextStats | null>(null)
  const [showCompressed, setShowCompressed] = useState(false)
  const [compacting, setCompacting] = useState(false)

  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const loadOverview = useCallback(async () => {
    setLoadingOverview(true)
    try {
      setOverview(await api.generalAgentOverview())
      setOverviewError('')
    } catch (error) {
      setOverviewError(errorText(error))
    } finally {
      setLoadingOverview(false)
    }
  }, [])

  const loadSessions = useCallback(async () => {
    setLoadingSessions(true)
    try {
      setSessions(await api.generalSessions())
      setSessionError('')
    } catch (error) {
      setSessionError(errorText(error))
    } finally {
      setLoadingSessions(false)
    }
  }, [])

  useEffect(() => {
    void loadOverview()
    void loadSessions()
  }, [loadOverview, loadSessions])

  useEffect(() => {
    const element = scrollRef.current
    if (!element) return
    const nearBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 180
    if (nearBottom) element.scrollTop = element.scrollHeight
  }, [items])

  const canChat = Boolean(overview?.ready && overview.llm_configured && overview.memory.ok)

  const toolGroups = useMemo(() => {
    if (!overview) return [] as { group: string; label: string; tools: GeneralAgentTool[] }[]
    const grouped = new Map<string, GeneralAgentTool[]>()
    for (const tool of overview.tools) {
      const list = grouped.get(tool.group) ?? []
      list.push(tool)
      grouped.set(tool.group, list)
    }
    const groups: { group: string; label: string; tools: GeneralAgentTool[] }[] = []
    for (const group of TOOL_GROUP_ORDER) {
      const tools = grouped.get(group)
      if (tools?.length) groups.push({ group, label: tools[0].group_label, tools })
    }
    return groups
  }, [overview])

  const upsertSession = useCallback((session: AgentChatSession) => {
    setSessions((previous) => {
      const rest = previous.filter((item) => item.thread_id !== session.thread_id)
      return [session, ...rest].sort((a, b) => b.updated_at.localeCompare(a.updated_at))
    })
  }, [])

  const patchAssistant = useCallback(
    (id: string, updater: (item: AssistantItem) => AssistantItem) => {
      setItems((previous) =>
        previous.map((item) =>
          item.kind === 'assistant' && item.id === id ? updater(item) : item,
        ),
      )
    },
    [],
  )

  const openSession = useCallback(
    async (session: AgentChatSession) => {
      if (streaming) {
        toast.warning('正在执行，请先停止当前对话')
        return
      }
      setThreadId(session.thread_id)
      setLoadingMessages(true)
      try {
        const data = await api.generalSessionMessages(session.thread_id)
        setSummaryInfo(data.summary)
        setItems(toItems(data.items, showCompressed))
      } catch (error) {
        toast.error(`读取会话失败：${errorText(error)}`)
        setItems([])
      } finally {
        setLoadingMessages(false)
      }
    },
    [showCompressed, streaming, toast],
  )

  const reloadMessages = useCallback(
    async (id: string, includeCompressed: boolean) => {
      setLoadingMessages(true)
      try {
        const data = await api.generalSessionMessages(id)
        setSummaryInfo(data.summary)
        setItems(toItems(data.items, includeCompressed))
      } catch (error) {
        toast.error(`读取会话失败：${errorText(error)}`)
      } finally {
        setLoadingMessages(false)
      }
    },
    [toast],
  )

  const toggleCompressed = useCallback(async () => {
    if (!threadId || streaming) return
    const next = !showCompressed
    setShowCompressed(next)
    await reloadMessages(threadId, next)
  }, [reloadMessages, showCompressed, streaming, threadId])

  const compactHistory = useCallback(async () => {
    if (!threadId || streaming || compacting) return
    setCompacting(true)
    try {
      const result = await api.compactGeneralSession(threadId)
      toast.success(result.message)
      await reloadMessages(threadId, showCompressed)
      await loadSessions()
      void loadOverview()
    } catch (error) {
      toast.error(`压缩失败：${errorText(error)}`)
    } finally {
      setCompacting(false)
    }
  }, [
    compacting,
    loadOverview,
    loadSessions,
    reloadMessages,
    showCompressed,
    streaming,
    threadId,
    toast,
  ])

  const removeSession = useCallback(
    async (session: AgentChatSession) => {
      try {
        await api.deleteGeneralSession(session.thread_id)
        setSessions((previous) => previous.filter((item) => item.thread_id !== session.thread_id))
        if (session.thread_id === threadId) {
          setThreadId(newId())
          setItems([])
          setSummaryInfo(null)
          setLastContext(null)
          setShowCompressed(false)
        }
        toast.success('会话已从 MySQL 删除')
      } catch (error) {
        toast.error(`删除失败：${errorText(error)}`)
      }
    },
    [threadId, toast],
  )

  const newSession = () => {
    abortRef.current?.abort()
    setThreadId(newId())
    setItems([])
    setInput('')
    setSummaryInfo(null)
    setLastContext(null)
    setShowCompressed(false)
  }

  const send = useCallback(
    async (text: string) => {
      const content = text.trim()
      if (!content || streaming || !canChat) return

      const currentThread = threadId || newId()
      if (!threadId) setThreadId(currentThread)
      const assistantId = newId()
      setItems((previous) => [
        ...previous,
        { kind: 'user', id: newId(), content },
        { kind: 'assistant', id: assistantId, content: '', status: 'loading', tools: [] },
      ])
      setInput('')
      setStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      const handlers: AgentChatStreamHandlers = {
        onStage: (stage, message) =>
          patchAssistant(assistantId, (item) => ({
            ...item,
            stage: message || stage,
          })),
        onSession: (session) => upsertSession(session),
        onToolCall: (call) =>
          patchAssistant(assistantId, (item) => ({
            ...item,
            stage: undefined,
            tools: [
              ...item.tools,
              {
                id: call.id,
                name: call.name,
                args: call.args,
                status: 'running',
                result: '',
                elapsedMs: null,
              },
            ],
          })),
        onToolResult: (result) =>
          patchAssistant(assistantId, (item) => ({
            ...item,
            stage: undefined,
            tools: item.tools.map((tool) =>
              tool.id === result.id
                ? {
                    ...tool,
                    status: result.status === 'error' ? 'error' : 'ok',
                    result: result.content,
                    elapsedMs: result.elapsed_ms,
                  }
                : tool,
            ),
          })),
        onToken: (token) =>
          patchAssistant(assistantId, (item) => ({
            ...item,
            content: item.content + token,
            status: 'streaming',
            stage: undefined,
          })),
        onDone: ({ answer, session, messages, context, summary }) => {
          upsertSession(session)
          setLastContext(context)
          setSummaryInfo(summary)
          if (messages.length) {
            // 用 MySQL 里落库的消息替换实时占位，刷新后与回放看到的内容完全一致。
            setItems((previous) => [
              ...previous.filter((item) => item.id !== assistantId),
              ...toItems(messages, showCompressed),
            ])
          } else {
            patchAssistant(assistantId, (item) => ({
              ...item,
              content: item.content || answer,
              status: 'done',
              stage: undefined,
            }))
          }
          if (summary) {
            // 自动压缩后，旧消息默认折叠；展开状态下标记为「已压缩」。
            const until = summary.until_seq
            setItems((previous) =>
              previous.flatMap((item) => {
                if (item.seq != null && item.seq <= until) {
                  return showCompressed ? [{ ...item, compressed: true }] : []
                }
                return [item]
              }),
            )
          }
        },
        onError: (detail) =>
          patchAssistant(assistantId, (item) => ({ ...item, status: 'error', error: detail })),
      }

      try {
        await api.generalChatStream(content, currentThread, handlers, controller.signal)
        if (controller.signal.aborted) {
          patchAssistant(assistantId, (item) => ({ ...item, status: 'stopped' }))
        } else {
          patchAssistant(assistantId, (item) =>
            item.status === 'loading' || item.status === 'streaming'
              ? { ...item, status: 'done' }
              : item,
          )
        }
      } catch (error) {
        if (controller.signal.aborted) {
          patchAssistant(assistantId, (item) => ({ ...item, status: 'stopped' }))
        } else {
          patchAssistant(assistantId, (item) => ({
            ...item,
            status: 'error',
            error: errorText(error),
          }))
        }
      } finally {
        setStreaming(false)
        abortRef.current = null
        void loadSessions()
      }
    },
    [canChat, loadSessions, patchAssistant, showCompressed, streaming, threadId, upsertSession],
  )

  const stop = () => {
    abortRef.current?.abort()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey) return
    if (event.nativeEvent.isComposing) return
    event.preventDefault()
    if (!streaming && canChat) void send(input)
  }

  const refresh = () => {
    void loadOverview()
    void loadSessions()
  }

  const showWelcome = items.length === 0 && !loadingMessages

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <Typography.Title level={4} style={{ margin: 0 }}>
            通用 Agent
          </Typography.Title>
          <Typography.Text type="secondary">
            LangChain ReAct 智能体 · 接入全部工具 · 会话记忆保存到 MySQL
          </Typography.Text>
        </div>
        <Space wrap>
          {overview && <Tag color="geekblue">{overview.llm_model}</Tag>}
          {overview && (
            <Tag color={overview.memory.ok ? 'green' : 'red'} icon={<DatabaseOutlined />}>
              MySQL 记忆{overview.memory.ok ? '' : '不可用'}
            </Tag>
          )}
          {overview && (
            <Tag color="cyan" icon={<CompressOutlined />}>
              上下文{' '}
              {lastContext
                ? `${formatTokens(lastContext.estimated_tokens)} / ${formatTokens(
                    lastContext.history_budget_tokens,
                  )}`
                : `预算 ${formatTokens(overview.context.history_budget_tokens)}`}
            </Tag>
          )}
          {summaryInfo && <Tag color="gold">已压缩 {summaryInfo.source_messages} 条</Tag>}
          {lastContext?.summary_error && (
            <Tooltip title={lastContext.summary_error}>
              <Tag color="orange">摘要降级</Tag>
            </Tooltip>
          )}
          {overview && <Tag color="purple">工具 {overview.tools.length}</Tag>}
          <Tooltip title="把更早的历史合并进摘要（原始消息仍保留在 MySQL）">
            <Button
              icon={<CompressOutlined />}
              loading={compacting}
              disabled={
                !canChat ||
                !threadId ||
                streaming ||
                !overview?.memory.summary_table_ready ||
                items.length === 0
              }
              onClick={() => void compactHistory()}
            >
              压缩历史
            </Button>
          </Tooltip>
          <Tooltip title="查看全部工具与说明">
            <Button icon={<ToolOutlined />} onClick={() => setToolsOpen(true)}>
              工具清单
            </Button>
          </Tooltip>
          <Tooltip title="重新读取模型、工具与会话记忆状态">
            <Button icon={<ReloadOutlined />} loading={loadingOverview} onClick={refresh}>
              刷新
            </Button>
          </Tooltip>
          <Button icon={<ThunderboltOutlined />} onClick={newSession} disabled={!canChat}>
            新会话
          </Button>
        </Space>
      </div>

      {overviewError && (
        <Alert
          type="error"
          showIcon
          message="无法读取通用 Agent 状态"
          description={`${overviewError} 请确认后端服务已启动后点「刷新」。`}
        />
      )}
      {overview && overview.warnings.length > 0 && (
        <Alert
          type="warning"
          showIcon
          message="配置提示"
          description={
            <ul className="agent-warning-list">
              {overview.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          }
        />
      )}

      <div className="agent-chat-layout">
        <aside className="agent-sessions">
          <div className="agent-sessions-head">
            <Space size={6}>
              <DatabaseOutlined />
              <Typography.Text strong>会话记忆</Typography.Text>
            </Space>
            <Tooltip title="刷新会话列表">
              <Button
                type="text"
                size="small"
                icon={<ReloadOutlined />}
                loading={loadingSessions}
                aria-label="刷新会话列表"
                onClick={() => void loadSessions()}
              />
            </Tooltip>
          </div>
          <div className="agent-sessions-list">
            {sessionError && (
              <Alert
                type="error"
                showIcon
                message="读取会话失败"
                description={sessionError}
                style={{ margin: 8 }}
              />
            )}
            {loadingSessions && sessions.length === 0 ? (
              <div className="agent-sessions-empty">
                <Spin />
              </div>
            ) : sessions.length === 0 ? (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="还没有会话，发一条消息就会存进 MySQL"
              />
            ) : (
              sessions.map((session) => (
                <div
                  key={session.thread_id}
                  className={`agent-session-item${
                    session.thread_id === threadId ? ' active' : ''
                  }`}
                  onClick={() => void openSession(session)}
                >
                  <div className="agent-session-title" title={session.title}>
                    {session.title}
                  </div>
                  <div className="agent-session-meta">
                    <span>{formatTime(session.updated_at)}</span>
                    <span>{session.message_count} 条</span>
                  </div>
                  <div className="agent-session-actions">
                    <Popconfirm
                      title="删除这个会话？"
                      description="会话与全部消息会从 MySQL 删除。"
                      okText="删除"
                      cancelText="取消"
                      onConfirm={() => void removeSession(session)}
                    >
                      <Button
                        type="text"
                        size="small"
                        danger
                        icon={<DeleteOutlined />}
                        aria-label="删除会话"
                        onClick={(event) => event.stopPropagation()}
                      />
                    </Popconfirm>
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="agent-sessions-foot">
            {overview ? overview.memory.tables.join(' · ') : 'MySQL 会话记忆'}
          </div>
        </aside>

        <div className="chat-panel panel">
          <div className="chat-scroll" ref={scrollRef}>
            {summaryInfo && (
              <div className="agent-summary-banner">
                <Space size={8} wrap>
                  <CompressOutlined />
                  <Typography.Text strong>更早的历史已压缩</Typography.Text>
                  <Typography.Text type="secondary">
                    覆盖 {summaryInfo.source_messages} 条 · seq ≤ {summaryInfo.until_seq}
                    {summaryInfo.tokens ? ` · 摘要 ${summaryInfo.tokens} tokens` : ''}
                  </Typography.Text>
                  <Button type="link" size="small" onClick={() => void toggleCompressed()}>
                    {showCompressed ? '隐藏已压缩消息' : '查看已压缩消息'}
                  </Button>
                </Space>
                <Collapse
                  size="small"
                  ghost
                  items={[
                    {
                      key: 'summary',
                      label: '查看摘要内容',
                      children: <pre className="agent-summary-text">{summaryInfo.summary}</pre>,
                    },
                  ]}
                />
              </div>
            )}
            {loadingMessages ? (
              <div className="agent-chat-loading">
                <Spin />
              </div>
            ) : showWelcome ? (
              <div className="chat-welcome">
                <div className="chat-welcome-icon">
                  <ThunderboltOutlined />
                </div>
                <Typography.Title level={4}>想让 Agent 做什么？</Typography.Title>
                <Typography.Paragraph type="secondary" style={{ maxWidth: 680 }}>
                  它会自己判断该用哪个工具：知识库检索、读写文件、执行命令、联网搜索、抓取网页。
                  每一轮对话都会写入 MySQL（tb_agent_chat_session / tb_agent_chat_message），
                  刷新页面或重启服务后还能接着聊。
                </Typography.Paragraph>
                <Space wrap className="chat-examples">
                  {EXAMPLES.map((example) => (
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
              items.map((item) => <MessageItem key={item.id} item={item} />)
            )}
          </div>

          <div className="chat-composer">
            <Input.TextArea
              autoSize={{ minRows: 2, maxRows: 6 }}
              value={input}
              disabled={!canChat}
              placeholder={
                canChat
                  ? '描述你想让 Agent 完成的任务，Enter 发送，Shift + Enter 换行'
                  : '模型或 MySQL 会话记忆未就绪，请先完成配置'
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
          <div className="chat-composer-hint agent-composer-hint">
            {threadId ? `会话 ${threadId.slice(0, 8)}` : '新会话'} · 记忆保存在 MySQL
            {overview
              ? ` · 上下文预算 ${formatTokens(overview.context.history_budget_tokens)} tokens`
              : ''}{' '}
            · 工作目录 {overview?.workspace ?? '—'}
          </div>
        </div>
      </div>

      <Drawer
        title="通用 Agent 可用工具"
        placement="right"
        width={480}
        open={toolsOpen}
        onClose={() => setToolsOpen(false)}
      >
        {overview ? (
          <>
            <Typography.Paragraph type="secondary" style={{ fontSize: 12 }}>
              全部工具都绑定在同一张 LangGraph 图上，由模型按任务自行选择；文件类工具只能在
              {overview.workspace} 内读写。最多连续 {overview.max_turns} 轮工具调用。
            </Typography.Paragraph>
            <Collapse
              defaultActiveKey={toolGroups.map((group) => group.group)}
              items={toolGroups.map((group) => ({
                key: group.group,
                label: `${group.label}（${group.tools.length}）`,
                children: (
                  <Space direction="vertical" size={10} style={{ width: '100%' }}>
                    {group.tools.map((tool) => (
                      <div key={tool.name} className="agent-tool-doc">
                        <Tag color="geekblue">{tool.name}</Tag>
                        <div className="agent-tool-doc-desc">{tool.description}</div>
                      </div>
                    ))}
                  </Space>
                ),
              }))}
            />
          </>
        ) : (
          <Empty description="还没有读取到工具清单" />
        )}
      </Drawer>
    </div>
  )
}
