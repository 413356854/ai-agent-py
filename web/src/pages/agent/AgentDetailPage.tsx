import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react'
import {
  CodeOutlined,
  CopyOutlined,
  DownloadOutlined,
  EditOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  UploadOutlined,
} from '@ant-design/icons'
import {
  Alert,
  App,
  Button,
  Card,
  Collapse,
  Descriptions,
  Modal,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tabs,
  Tag,
  Timeline,
  Typography,
  Upload,
} from 'antd'
import type { TableProps } from 'antd'
import { useParams } from 'react-router-dom'
import { ApiError, api } from '../../api/client'
import { ClaimDataCard } from '../../components/ClaimDataCard'
import { ParamEditor } from '../../components/ParamEditor'
import type {
  AgentConfig,
  AgentDetail,
  AgentFinding,
  AgentGraphNode,
  AgentRule,
  AgentRunResult,
  AgentSample,
  ClaimData,
} from '../../types'

/** mermaid 体积较大，只有打开流程图时才加载。 */
const MermaidDiagram = lazy(() =>
  import('../../components/MermaidDiagram').then((module) => ({ default: module.MermaidDiagram })),
)

const STATUS_COLOR: Record<string, string> = { 通过: 'green', 退回: 'red', 转人工: 'gold' }
const SEVERITY_COLOR: Record<string, string> = {
  阻断: 'red',
  高: 'volcano',
  中: 'orange',
  警告: 'gold',
  提示: 'blue',
}
const KIND_LABEL: Record<string, string> = {
  node: '节点',
  checker: '检查器',
  agent: 'Agent',
  human: '人工',
}

/** Finding.evidence 的字段名 → 中文标签（展开行用）。 */
const EVIDENCE_LABEL: Record<string, string> = {
  number: '发票号',
  numbers: '连号发票',
  date: '开票日期',
  period: '报销期间',
  category: '费用类型',
  kind: '票种',
  amount: '金额',
  tax: '税额',
  total: '合计',
  sum: '发票合计',
  diff: '差额',
  nightly: '每晚金额',
  tier: '城市档',
  cap: '标准上限',
  per_head: '人均',
  items: '票据项目',
  buyer: '购买方',
  missing: '缺失字段',
  approver: '审批人',
}

interface ParamRow {
  key: string
  value: string
  usedBy: string[]
  status: string
}

/** 参数值可能是对象（如住宿上限按城市分级）或数组（如允许票种）。 */
function formatParamValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (Array.isArray(value)) return value.join(' / ')
  if (typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, nested]) => `${key} ${String(nested)}`)
      .join(' ｜ ')
  }
  return String(value)
}

function errorText(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return String(error)
}

/** 报销审核 Agent 详情：流程图（LangGraph 导出）、试跑、规则库。 */
export function AgentDetailPage() {
  const { agentId = '' } = useParams<{ agentId: string }>()
  const { message } = App.useApp()

  const [detail, setDetail] = useState<AgentDetail | null>(null)
  const [samples, setSamples] = useState<AgentSample[]>([])
  const [config, setConfig] = useState<AgentConfig | null>(null)
  const [sampleId, setSampleId] = useState('')
  const [uploaded, setUploaded] = useState<{ claim: ClaimData; filename: string; warnings: string[] } | null>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<AgentRunResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [sourceOpen, setSourceOpen] = useState(false)
  const [reloadKey, setReloadKey] = useState(0)
  const [editorOpen, setEditorOpen] = useState(false)
  const [draft, setDraft] = useState<Record<string, unknown> | null>(null)
  const [saving, setSaving] = useState(false)

  const load = useCallback(async () => {
    if (!agentId) return
    setLoading(true)
    try {
      const [nextDetail, nextSamples, nextConfig] = await Promise.all([
        api.agent(agentId),
        api.agentSamples(agentId),
        api.agentConfig(agentId),
      ])
      setDetail(nextDetail)
      setSamples(nextSamples)
      setConfig(nextConfig)
      setSampleId((current) => current || nextSamples[0]?.sample_id || '')
      setError('')
    } catch (err) {
      setError(errorText(err))
    } finally {
      setLoading(false)
    }
  }, [agentId])

  useEffect(() => {
    void load()
  }, [load, reloadKey])

  const run = useCallback(async () => {
    if (!agentId) return
    if (!uploaded && !sampleId) return
    setRunning(true)
    try {
      const next = uploaded
        ? await api.runAgentClaim(agentId, uploaded.claim as unknown as Record<string, unknown>)
        : await api.runAgent(agentId, sampleId)
      setResult(next)
    } catch (err) {
      message.error(errorText(err))
    } finally {
      setRunning(false)
    }
  }, [agentId, message, sampleId, uploaded])

  const handleUpload = useCallback(
    async (file: File) => {
      if (!agentId) return
      setUploading(true)
      try {
        const parsed = await api.parseUpload(agentId, file)
        setUploaded({ claim: parsed.claim, filename: parsed.filename, warnings: parsed.warnings })
        setSampleId('')
        setResult(null)
        message.success(`已解析「${parsed.filename}」，点击「开始试跑」查看结果`)
      } catch (err) {
        message.error(errorText(err))
      } finally {
        setUploading(false)
      }
    },
    [agentId, message],
  )

  const copyMermaid = useCallback(async () => {
    if (!detail) return
    try {
      await navigator.clipboard.writeText(detail.graph.mermaid)
      message.success('Mermaid 源码已复制')
    } catch {
      message.error('复制失败，请手动从源码弹窗复制')
    }
  }, [detail, message])

  const openEditor = useCallback(() => {
    setDraft(JSON.parse(JSON.stringify(config?.params ?? {})) as Record<string, unknown>)
    setEditorOpen(true)
  }, [config])

  const saveParams = useCallback(async () => {
    if (!agentId || !draft) return
    setSaving(true)
    try {
      setConfig(await api.updateAgentParams(agentId, draft))
      setEditorOpen(false)
      setResult(null)
      message.success('参数已保存并生效，试跑结果已清空，请重新试跑')
    } catch (err) {
      message.error(errorText(err))
    } finally {
      setSaving(false)
    }
  }, [agentId, draft, message])

  const reloadConfig = useCallback(async () => {
    if (!agentId) return
    try {
      setConfig(await api.reloadAgentConfig(agentId))
      setResult(null)
      message.success('已重新读取参数文件')
    } catch (err) {
      message.error(errorText(err))
    }
  }, [agentId, message])

  const nodeColumns: TableProps<AgentGraphNode>['columns'] = [
    {
      title: '门禁',
      dataIndex: 'gate',
      width: 88,
      render: (value: string) => <Tag color="geekblue">{value || '—'}</Tag>,
    },
    { title: '节点', dataIndex: 'label', width: 140 },
    {
      title: '类型',
      dataIndex: 'kind',
      width: 100,
      render: (value: string) => <Tag>{KIND_LABEL[value] ?? value}</Tag>,
    },
    { title: '说明', dataIndex: 'description' },
  ]

  const findingColumns: TableProps<AgentFinding>['columns'] = [
    { title: '规则', dataIndex: 'rule_id', width: 80 },
    {
      title: '级别',
      dataIndex: 'severity',
      width: 80,
      render: (value: string) => <Tag color={SEVERITY_COLOR[value] ?? 'default'}>{value}</Tag>,
    },
    { title: '类别', dataIndex: 'category', width: 110 },
    { title: '说明', dataIndex: 'message' },
    {
      title: '依据',
      dataIndex: 'citation',
      width: 240,
      render: (value: string) => <Typography.Text type="secondary">{value}</Typography.Text>,
    },
  ]

  const ruleColumns: TableProps<AgentRule>['columns'] = [
    { title: '编号', dataIndex: 'id', width: 80 },
    { title: '规则', dataIndex: 'name', width: 200 },
    { title: '类别', dataIndex: 'category', width: 110 },
    {
      title: '级别',
      dataIndex: 'severity',
      width: 76,
      render: (value: string) => <Tag color={SEVERITY_COLOR[value] ?? 'default'}>{value}</Tag>,
    },
    {
      title: '阶段',
      dataIndex: 'stage',
      width: 90,
      render: (value: string) => <Tag>{value}</Tag>,
    },
    {
      title: '使用参数',
      dataIndex: 'params',
      width: 190,
      render: (value: string[] | undefined) =>
        value && value.length ? (
          <Space size={4} wrap>
            {value.map((item) => (
              <Tag key={item} color="blue">
                {item}
              </Tag>
            ))}
          </Space>
        ) : (
          <Typography.Text type="secondary">—</Typography.Text>
        ),
    },
    { title: '依据', dataIndex: 'citation' },
  ]

  const paramColumns: TableProps<ParamRow>['columns'] = [
    {
      title: '参数键',
      dataIndex: 'key',
      width: 230,
      render: (value: string) => <Typography.Text code>{value}</Typography.Text>,
    },
    { title: '取值', dataIndex: 'value' },
    {
      title: '使用规则',
      dataIndex: 'usedBy',
      width: 170,
      render: (value: string[]) =>
        value.length ? (
          <Space size={4} wrap>
            {value.map((id) => (
              <Tag key={id}>{id}</Tag>
            ))}
          </Space>
        ) : (
          <Typography.Text type="secondary">—</Typography.Text>
        ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 92,
      render: (value: string) => <Tag color={value === '占位' ? 'orange' : 'green'}>{value}</Tag>,
    },
  ]

  const paramRows = useMemo<ParamRow[]>(() => {
    if (!config) return []
    return Object.entries(config.params).map(([key, value]) => ({
      key,
      value: formatParamValue(value),
      usedBy: config.rules.filter((rule) => (rule.params ?? []).includes(key)).map((rule) => rule.id),
      status: config.param_status === 'placeholder' ? '占位' : '已确认',
    }))
  }, [config])

  const traceItems = useMemo(
    () =>
      (result?.trace ?? []).map((event) => ({
        color: 'blue',
        children: (
          <Space size={8} wrap>
            <Tag color="geekblue">{event.gate || event.node}</Tag>
            <Typography.Text strong>{event.label}</Typography.Text>
            <Typography.Text type="secondary">{event.summary}</Typography.Text>
            <Typography.Text type="secondary">{event.ms} ms</Typography.Text>
          </Space>
        ),
      })),
    [result],
  )

  if (loading) {
    return (
      <div style={{ padding: 64, textAlign: 'center' }}>
        <Spin />
      </div>
    )
  }

  if (error || !detail) {
    return <Alert type="error" showIcon message="加载失败" description={error || '应用不存在'} />
  }

  const graph = detail.graph
  const selectedSample = samples.find((item) => item.sample_id === sampleId)
  const activeClaim = uploaded?.claim ?? selectedSample?.claim

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 24, flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 420px' }}>
            <Space size={8} wrap>
              <Typography.Title level={4} style={{ margin: 0 }}>
                {detail.name}
              </Typography.Title>
              <Tag color="green">{detail.status}</Tag>
              <Tag>v{detail.version}</Tag>
            </Space>
            <Typography.Paragraph type="secondary" style={{ margin: '10px 0 12px' }}>
              {detail.summary}
            </Typography.Paragraph>
            <Space size={4} wrap>
              {detail.tags.map((tag) => (
                <Tag key={tag}>{tag}</Tag>
              ))}
            </Space>
          </div>
          <Space size={32}>
            <Statistic title="节点" value={graph.node_count} />
            <Statistic title="流转边" value={graph.edge_count} />
            <Statistic title="规则" value={graph.rule_count} />
            <Statistic
              title="参数口径"
              value={graph.param_status === 'placeholder' ? '占位' : '已确认'}
              valueStyle={graph.param_status === 'placeholder' ? { color: '#d48806' } : undefined}
            />
          </Space>
        </div>
      </Card>

      <Tabs
        defaultActiveKey="graph"
        items={[
          {
            key: 'graph',
            label: '流程图',
            children: (
              <Space direction="vertical" size={16} style={{ width: '100%' }}>
                <Card
                  title="LangGraph 流程图"
                  extra={
                    <Space>
                      <Typography.Text type="secondary">
                        get_graph().draw_mermaid()
                      </Typography.Text>
                      <Button icon={<ReloadOutlined />} onClick={() => setReloadKey((v) => v + 1)}>
                        刷新
                      </Button>
                      <Button icon={<CodeOutlined />} onClick={() => setSourceOpen(true)}>
                        Mermaid 源码
                      </Button>
                      <Button icon={<CopyOutlined />} onClick={() => void copyMermaid()}>
                        复制
                      </Button>
                    </Space>
                  }
                >
                  <Suspense
                    fallback={
                      <div style={{ padding: 48, textAlign: 'center' }}>
                        <Spin />
                      </div>
                    }
                  >
                    <MermaidDiagram code={graph.mermaid} />
                  </Suspense>
                </Card>

                <Card title="节点清单" size="small">
                  <Table<AgentGraphNode>
                    rowKey="id"
                    size="small"
                    pagination={false}
                    columns={nodeColumns}
                    dataSource={graph.nodes}
                  />
                </Card>
              </Space>
            ),
          },
          {
            key: 'run',
            label: '试跑',
            children: (
              <Space direction="vertical" size={16} style={{ width: '100%' }}>
                <Alert
                  type="info"
                  showIcon
                  message="内置样例为合成数据"
                  description="语义疑点（R4）只会「转人工」，不会自动退回；硬规则命中才「退回」。"
                />
                <Card size="small">
                  <Space wrap>
                    <Select
                      style={{ width: 340 }}
                      value={sampleId || undefined}
                      onChange={(value) => {
                        setSampleId(value)
                        setUploaded(null)
                        setResult(null)
                      }}
                      placeholder="选择内置样例"
                      options={samples.map((item) => ({
                        value: item.sample_id,
                        label: `${item.sample_id} · ${item.title}（预期：${item.expected}）`,
                      }))}
                    />
                    <Button
                      icon={<DownloadOutlined />}
                      disabled={!selectedSample}
                      onClick={() =>
                        selectedSample &&
                        window.open(api.agentSampleDownloadUrl(agentId, selectedSample.sample_id), '_blank')
                      }
                    >
                      下载样板
                    </Button>
                    <Button
                      type="link"
                      onClick={() => window.open(api.agentTemplateUrl(agentId), '_blank')}
                    >
                      空白模板
                    </Button>
                    <Upload
                      accept=".xlsx,.json"
                      showUploadList={false}
                      beforeUpload={(file) => {
                        void handleUpload(file as File)
                        return false
                      }}
                    >
                      <Button icon={<UploadOutlined />} loading={uploading}>
                        上传单据
                      </Button>
                    </Upload>
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      loading={running}
                      disabled={!sampleId && !uploaded}
                      onClick={() => void run()}
                    >
                      开始试跑
                    </Button>
                  </Space>
                  {uploaded ? (
                    <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
                      已解析上传文件「{uploaded.filename}」，点「开始试跑」用这份单据审核；重新选样例会清除上传。
                    </Typography.Paragraph>
                  ) : (
                    sampleId && (
                      <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
                        {selectedSample?.note}
                      </Typography.Paragraph>
                    )
                  )}
                </Card>

                {uploaded && uploaded.warnings.length > 0 && (
                  <Alert
                    type="warning"
                    showIcon
                    message="上传解析提示"
                    description={
                      <ul style={{ margin: 0, paddingLeft: 18 }}>
                        {uploaded.warnings.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    }
                  />
                )}

                {activeClaim && (
                  <ClaimDataCard
                    claim={activeClaim}
                    badge={uploaded ? '上传' : selectedSample?.sample_id}
                    title={uploaded ? uploaded.filename : selectedSample?.title}
                    expected={uploaded ? undefined : selectedSample?.expected}
                    findings={
                      result && result.claim_id === activeClaim.claim_id ? result.findings : []
                    }
                    extra={
                      uploaded ? (
                        <Button
                          size="small"
                          onClick={() => {
                            setUploaded(null)
                            setResult(null)
                          }}
                        >
                          清除上传
                        </Button>
                      ) : undefined
                    }
                  />
                )}

                {result && (
                  <Card
                    title={
                      <Space>
                        <span>试跑结果</span>
                        <Tag color={STATUS_COLOR[result.status] ?? 'default'}>{result.status}</Tag>
                        {selectedSample && (
                          <Tag
                            color={selectedSample.expected === result.status ? 'green' : 'red'}
                          >
                            {selectedSample.expected === result.status
                              ? '符合预期'
                              : `预期 ${selectedSample.expected}`}
                          </Tag>
                        )}
                        <Typography.Text type="secondary">
                          {result.claim_id} · {result.elapsed_ms} ms
                        </Typography.Text>
                      </Space>
                    }
                  >
                    <Alert
                      type={result.status === '退回' ? 'error' : result.status === '通过' ? 'success' : 'warning'}
                      showIcon
                      message={result.status}
                      description={result.decision.reason}
                      style={{ marginBottom: 16 }}
                    />
                    <Space size={48} style={{ marginBottom: 16 }}>
                      <Statistic title="判定置信度" value={result.confidence * 100} suffix="%" precision={0} />
                      <Statistic title="命中 Finding" value={result.findings.length} />
                      <Statistic title="耗时" value={result.elapsed_ms} suffix="ms" />
                    </Space>

                    <Table<AgentFinding>
                      rowKey={(row) => `${row.rule_id}-${row.message}`}
                      size="small"
                      pagination={false}
                      columns={findingColumns}
                      dataSource={result.findings}
                      locale={{ emptyText: '未命中任何规则' }}
                      style={{ marginBottom: 12 }}
                      expandable={{
                        rowExpandable: (row) => Object.keys(row.evidence ?? {}).length > 0,
                        expandedRowRender: (row) => (
                          <Descriptions
                            size="small"
                            column={2}
                            bordered
                            items={Object.entries(row.evidence ?? {}).map(([key, value]) => ({
                              key,
                              label: EVIDENCE_LABEL[key] ?? key,
                              children: String(value),
                            }))}
                          />
                        ),
                      }}
                    />

                    <Collapse
                      ghost
                      style={{ marginBottom: 20 }}
                      items={[
                        {
                          key: 'params',
                          label: (
                            <Space>
                              <span>对比：本次判定使用的参数</span>
                              {graph.param_status === 'placeholder' && <Tag color="orange">占位，未经财务确认</Tag>}
                            </Space>
                          ),
                          children: <ParamList params={config?.params ?? {}} />,
                        },
                      ]}
                    />

                    <Typography.Title level={5}>节点轨迹</Typography.Title>
                    <Timeline items={traceItems} />
                  </Card>
                )}
              </Space>
            ),
          },
          {
            key: 'rules',
            label: '规则库与参数',
            children: (
              <Space direction="vertical" size={16} style={{ width: '100%' }}>
                {config?.param_status === 'placeholder' && (
                  <Alert
                    type="warning"
                    showIcon
                    message="参数为占位值，未经财务确认"
                    description={
                      <>
                        {config.param_note}
                        <br />
                        参数文件：
                        <Typography.Text code>{config.param_file}</Typography.Text>
                        （版本 {config.param_version || '—'}）
                      </>
                    }
                  />
                )}
                <Card title={`规则库（${config?.rules.length ?? 0} 条）`} size="small">
                  <Table<AgentRule>
                    rowKey="id"
                    size="small"
                    pagination={false}
                    columns={ruleColumns}
                    dataSource={config?.rules ?? []}
                  />
                </Card>
                <Card
                  size="small"
                  title={`参数库（${paramRows.length} 项）`}
                  extra={
                    <Space size={8}>
                      <Tag color={config?.param_status === 'placeholder' ? 'orange' : 'green'}>
                        {config?.param_status === 'placeholder' ? '占位' : '已确认'}
                      </Tag>
                      <Button size="small" icon={<EditOutlined />} onClick={openEditor}>
                        编辑参数
                      </Button>
                      <Button
                        size="small"
                        icon={<ReloadOutlined />}
                        onClick={() => void reloadConfig()}
                      >
                        重新加载
                      </Button>
                    </Space>
                  }
                >
                  <Table<ParamRow>
                    rowKey="key"
                    size="small"
                    pagination={false}
                    columns={paramColumns}
                    dataSource={paramRows}
                  />
                  <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
                    参数文件：<Typography.Text code>{config?.param_file}</Typography.Text>
                    （版本 {config?.param_version || '—'}
                    {config?.param_updated_at
                      ? `，最近保存 ${config.param_updated_at.replace('T', ' ')}`
                      : ''}
                    ）。保存后立即落盘并刷新进程缓存，不需要重启服务。
                  </Typography.Paragraph>
                </Card>
              </Space>
            ),
          },
        ]}
      />

      <Modal
        title="Mermaid 源码（LangGraph 原始导出）"
        open={sourceOpen}
        onCancel={() => setSourceOpen(false)}
        footer={null}
        width={900}
      >
        <Typography.Paragraph type="secondary">
          页面上的节点标签与条件边文字经过中文替换与合法化处理，拓扑与原导出完全一致。
        </Typography.Paragraph>
        <pre className="agent-source">{graph.mermaid_source}</pre>
      </Modal>

      <Modal
        title="编辑参数"
        open={editorOpen}
        onCancel={() => setEditorOpen(false)}
        onOk={() => void saveParams()}
        okText="保存并生效"
        cancelText="取消"
        confirmLoading={saving}
        width={680}
      >
        <Alert
          type="info"
          showIcon
          message="保存后立即落盘并刷新缓存"
          description="参数变化会影响审核结论；保存成功后试跑结果会清空，请重新试跑验证。"
          style={{ marginBottom: 16 }}
        />
        {draft && <ParamEditor value={draft} onChange={setDraft} />}
      </Modal>
    </Space>
  )
}

/** 把参数库里的关键阈值渲染成人话，便于和命中结果对照。 */
function ParamList({ params }: { params: Record<string, unknown> }) {
  const rows: { label: string; value: string }[] = []

  const lodging = params.lodging_cap_per_night as Record<string, number> | undefined
  if (lodging) {
    rows.push({
      label: '住宿上限（元/晚）',
      value: Object.entries(lodging)
        .map(([tier, cap]) => `${tier} ${cap}`)
        .join(' ｜ '),
    })
  }
  if (params.entertainment_per_head !== undefined) {
    rows.push({ label: '招待人均上限（元）', value: String(params.entertainment_per_head) })
  }
  const kinds = params.allowed_kinds as Record<string, string[]> | undefined
  if (kinds) {
    rows.push({
      label: '允许票种',
      value: Object.entries(kinds)
        .map(([category, list]) => `${category}：${list.join(' / ')}`)
        .join('；'),
    })
  }

  if (rows.length === 0) {
    return <pre className="agent-params">{JSON.stringify(params, null, 2)}</pre>
  }

  return (
    <Descriptions
      size="small"
      bordered
      column={1}
      items={rows.map((row, index) => ({ key: index, label: row.label, children: row.value }))}
    />
  )
}
