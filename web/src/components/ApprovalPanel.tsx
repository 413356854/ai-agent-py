import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  EyeOutlined,
  ReloadOutlined,
  SendOutlined,
  StopOutlined,
} from '@ant-design/icons'
import {
  Alert,
  App,
  Button,
  Card,
  Descriptions,
  Empty,
  Input,
  Modal,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Timeline,
  Typography,
} from 'antd'
import type { TableProps } from 'antd'
import { ApiError, api } from '../api/client'
import type {
  AgentApprovalRequest,
  AgentFinding,
  AgentRun,
  AgentSample,
  AgentTraceEvent,
} from '../types'

/** 运行状态 → 中文与颜色（与后端 runs/models.py 的 RunStatus 对齐）。 */
const RUN_STATUS: Record<string, { label: string; color: string }> = {
  running: { label: '执行中', color: 'processing' },
  pending: { label: '待审批', color: 'gold' },
  escalated: { label: '已升级', color: 'volcano' },
  approved: { label: '已通过', color: 'green' },
  rejected: { label: '已拒绝', color: 'red' },
  timeout: { label: '超时拒绝', color: 'red' },
  cancelled: { label: '已取消', color: 'default' },
  completed: { label: '已完成', color: 'blue' },
  failed: { label: '失败', color: 'red' },
}

const RISK_COLOR: Record<string, string> = { low: 'green', medium: 'gold', high: 'red' }
const SEVERITY_COLOR: Record<string, string> = {
  阻断: 'red',
  高: 'volcano',
  中: 'orange',
  警告: 'gold',
  提示: 'blue',
}

function errorText(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return String(error)
}

function formatTime(value: string | null): string {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

function runStatus(status: string) {
  return RUN_STATUS[status] ?? { label: status, color: 'default' }
}

/** 运行详情：审批内容 + 结论 + 节点轨迹 + 流转记录。 */
function RunDetail({ run }: { run: AgentRun }) {
  const request = run.approval
  const result = run.result
  const trace = (result?.trace ?? []) as AgentTraceEvent[]
  const findings = (result?.findings ?? []) as AgentFinding[]

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Descriptions
        size="small"
        column={2}
        bordered
        items={[
          { key: 'run', label: '运行', children: <Typography.Text code>{run.run_id}</Typography.Text> },
          {
            key: 'status',
            label: '状态',
            children: (
              <Tag color={runStatus(run.status).color}>{runStatus(run.status).label}</Tag>
            ),
          },
          { key: 'claim', label: '单据', children: run.title || '—' },
          { key: 'created', label: '提交时间', children: formatTime(run.created_at) },
          { key: 'expires', label: '审批截止', children: formatTime(run.expires_at) },
          { key: 'finished', label: '结束时间', children: formatTime(run.finished_at) },
          { key: 'approver', label: '审批人', children: run.approver || '—' },
          { key: 'comment', label: '审批意见', children: run.comment || '—' },
        ]}
      />

      {run.error && <Alert type="error" showIcon message="执行失败" description={run.error} />}

      {request && (
        <Card size="small" title="待审批内容">
          <Space direction="vertical" size={8} style={{ width: '100%' }}>
            <Typography.Text strong>{request.title}</Typography.Text>
            <Typography.Text type="secondary">{request.summary || '—'}</Typography.Text>
            <Space size={8} wrap>
              <Tag color={RISK_COLOR[request.risk] ?? 'default'}>风险：{request.risk}</Tag>
              <Tag>节点：{request.node || '—'}</Tag>
              {request.tool && <Tag color="geekblue">工具：{request.tool}</Tag>}
              {request.claim_id && <Tag>{request.claim_id}</Tag>}
            </Space>
            {Object.keys(request.args ?? {}).length > 0 && (
              <Descriptions
                size="small"
                column={2}
                items={Object.entries(request.args).map(([key, value]) => ({
                  key,
                  label: key,
                  children: String(value ?? '—'),
                }))}
              />
            )}
            {findings.length > 0 && (
              <Table<AgentFinding>
                rowKey={(row) => `${row.rule_id}-${row.message}`}
                size="small"
                pagination={false}
                dataSource={findings}
                columns={[
                  { title: '规则', dataIndex: 'rule_id', width: 80 },
                  {
                    title: '级别',
                    dataIndex: 'severity',
                    width: 76,
                    render: (value: string) => (
                      <Tag color={SEVERITY_COLOR[value] ?? 'default'}>{value}</Tag>
                    ),
                  },
                  { title: '说明', dataIndex: 'message' },
                ]}
              />
            )}
          </Space>
        </Card>
      )}

      {result && (
        <Card size="small" title="执行结果">
          <Space direction="vertical" size={8} style={{ width: '100%' }}>
            <Space size={16} wrap>
              <Tag color={result.status === '通过' ? 'green' : result.status === '退回' ? 'red' : 'gold'}>
                {result.status || '—'}
              </Tag>
              <Typography.Text type="secondary">{result.decision?.reason ?? ''}</Typography.Text>
            </Space>
            {result.approval && (
              <Typography.Text type="secondary">
                审批：{result.approval.approved ? '通过' : '拒绝'} · {result.approval.approver}
                {result.approval.timeout ? '（超时自动拒绝）' : ''}
                {result.approval.comment ? ` · ${result.approval.comment}` : ''}
              </Typography.Text>
            )}
            {trace.length > 0 && (
              <Timeline
                items={trace.map((event) => ({
                  color: 'blue',
                  children: (
                    <Space size={8} wrap>
                      <Tag color="geekblue">{event.gate || event.node}</Tag>
                      <Typography.Text strong>{event.label}</Typography.Text>
                      <Typography.Text type="secondary">{event.summary}</Typography.Text>
                      <Typography.Text type="secondary">{event.ms} ms</Typography.Text>
                    </Space>
                  ),
                }))}
              />
            )}
          </Space>
        </Card>
      )}

      {run.events.length > 0 && (
        <Card size="small" title="状态流转">
          <Timeline
            items={run.events.map((event, index) => ({
              key: `${event.at}-${index}`,
              color: runStatus(event.status).color,
              children: (
                <Space size={8} wrap>
                  <Tag color={runStatus(event.status).color}>{runStatus(event.status).label}</Tag>
                  <Typography.Text type="secondary">{formatTime(event.at)}</Typography.Text>
                  {event.by && <Typography.Text>{event.by}</Typography.Text>}
                  {event.note && <Typography.Text type="secondary">{event.note}</Typography.Text>}
                </Space>
              ),
            }))}
          />
        </Card>
      )}
    </Space>
  )
}

interface ApprovalPanelProps {
  agentId: string
  samples: AgentSample[]
}

/**
 * 人工审批台：提交一次正式运行（会停在 G8），审批通过 / 拒绝 / 撤销，并回看运行留痕。
 *
 * 列表每 15 秒刷新一次：后端在新的读请求上会做超时巡检，所以「已升级 / 超时拒绝」
 * 会自己出现，不需要手工点刷新。
 */
export function ApprovalPanel({ agentId, samples }: ApprovalPanelProps) {
  const { message } = App.useApp()
  const [runs, setRuns] = useState<AgentRun[]>([])
  const [loading, setLoading] = useState(false)
  const [sampleId, setSampleId] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [pendingDecision, setPendingDecision] = useState<{
    run: AgentRun
    kind: 'approve' | 'reject'
  } | null>(null)
  const [approver, setApprover] = useState('')
  const [comment, setComment] = useState('')
  const [working, setWorking] = useState('')
  const [detail, setDetail] = useState<AgentRun | null>(null)

  const load = useCallback(async () => {
    if (!agentId) return
    setLoading(true)
    try {
      setRuns(await api.agentRuns(agentId))
    } catch (err) {
      message.error(errorText(err))
    } finally {
      setLoading(false)
    }
  }, [agentId, message])

  useEffect(() => {
    void load()
    const timer = window.setInterval(() => {
      void load()
    }, 15000)
    return () => window.clearInterval(timer)
  }, [load])

  const waiting = useMemo(
    () => runs.filter((run) => run.status === 'pending' || run.status === 'escalated'),
    [runs],
  )

  const submit = useCallback(async () => {
    if (!sampleId) return
    setSubmitting(true)
    try {
      const run = await api.startAgentRun(agentId, { sample_id: sampleId })
      message.success(
        run.status === 'pending' ? '已停在 G8，等待人工审批' : `已直接执行完成：${run.result?.status ?? ''}`,
      )
      setDetail(run)
      setSampleId('')
      await load()
    } catch (err) {
      message.error(errorText(err))
    } finally {
      setSubmitting(false)
    }
  }, [agentId, approver, load, message, sampleId])

  const decide = useCallback(async () => {
    if (!pendingDecision) return
    const { run, kind } = pendingDecision
    setWorking(run.run_id)
    try {
      const updated = await api.resumeAgentRun(agentId, run.run_id, {
        decision: kind === 'approve' ? 'approve' : 'reject',
        approver: approver || undefined,
        comment: comment || undefined,
      })
      message.success(
        kind === 'approve'
          ? `已通过，结论：${updated.result?.status ?? '已继续执行'}`
          : '已拒绝，单据退回',
      )
      setPendingDecision(null)
      setComment('')
      setDetail(updated)
      await load()
    } catch (err) {
      message.error(errorText(err))
    } finally {
      setWorking('')
    }
  }, [agentId, approver, comment, load, message, pendingDecision])

  const cancelRun = useCallback(
    async (run: AgentRun) => {
      setWorking(run.run_id)
      try {
        const updated = await api.cancelAgentRun(agentId, run.run_id, {
          approver: approver || undefined,
          comment: comment || '页面撤销',
        })
        message.success('已撤销该运行')
        setDetail(updated)
        await load()
      } catch (err) {
        message.error(errorText(err))
      } finally {
        setWorking('')
      }
    },
    [agentId, approver, comment, load, message],
  )

  const columns: TableProps<AgentRun>['columns'] = [
    {
      title: '状态',
      dataIndex: 'status',
      width: 104,
      render: (value: string) => <Tag color={runStatus(value).color}>{runStatus(value).label}</Tag>,
    },
    {
      title: '单据',
      dataIndex: 'title',
      render: (value: string, row) => (
        <Space direction="vertical" size={0}>
          <Typography.Text>{value || '—'}</Typography.Text>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {row.run_id.slice(0, 8)}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: '待审批内容',
      dataIndex: 'approval',
      render: (value: AgentApprovalRequest | null) =>
        value ? (
          <Space size={4} wrap>
            <Tag color={RISK_COLOR[value.risk] ?? 'default'}>{value.risk}</Tag>
            <Typography.Text>{value.summary || value.title}</Typography.Text>
          </Space>
        ) : (
          <Typography.Text type="secondary">—</Typography.Text>
        ),
    },
    {
      title: '提交时间',
      dataIndex: 'created_at',
      width: 168,
      render: (value: string) => formatTime(value),
    },
    {
      title: '审批截止',
      dataIndex: 'expires_at',
      width: 168,
      render: (value: string | null) => formatTime(value),
    },
    {
      title: '审批人',
      dataIndex: 'approver',
      width: 104,
      render: (value: string | null) => value || '—',
    },
    {
      title: '操作',
      key: 'actions',
      width: 236,
      render: (_value, row) => (
        <Space size={4} wrap>
          {(row.status === 'pending' || row.status === 'escalated') && (
            <>
              <Button
                size="small"
                type="primary"
                icon={<CheckCircleOutlined />}
                loading={working === row.run_id}
                onClick={() => setPendingDecision({ run: row, kind: 'approve' })}
              >
                通过
              </Button>
              <Button
                size="small"
                danger
                icon={<CloseCircleOutlined />}
                onClick={() => setPendingDecision({ run: row, kind: 'reject' })}
              >
                拒绝
              </Button>
              <Button
                size="small"
                icon={<StopOutlined />}
                onClick={() => void cancelRun(row)}
              >
                撤销
              </Button>
            </>
          )}
          <Button size="small" icon={<EyeOutlined />} onClick={() => setDetail(row)}>
            详情
          </Button>
        </Space>
      ),
    },
  ]

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Alert
        type="info"
        showIcon
        message="人工审批（HITL）"
        description="提交后图会停在 G8「人工复核」并落检查点：审批通过继续归档，拒绝则退回；超时未处理会先升级、再自动拒绝。"
      />

      <Card size="small">
        <Space wrap align="center">
          <Statistic title="待审批" value={waiting.length} />
          <Statistic title="运行总数" value={runs.length} />
          <Select
            style={{ width: 340 }}
            value={sampleId || undefined}
            onChange={setSampleId}
            placeholder="选择要送审的内置样例"
            options={samples.map((item) => ({
              value: item.sample_id,
              label: `${item.sample_id} · ${item.title}（预期：${item.expected}）`,
            }))}
          />
          <Input
            style={{ width: 160 }}
            value={approver}
            onChange={(event) => setApprover(event.target.value)}
            placeholder="审批人（可选）"
            maxLength={64}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            loading={submitting}
            disabled={!sampleId}
            onClick={() => void submit()}
          >
            提交送审
          </Button>
          <Button icon={<ReloadOutlined />} loading={loading} onClick={() => void load()}>
            刷新
          </Button>
        </Space>
      </Card>

      <Card size="small" title="运行记录">
        <Table<AgentRun>
          rowKey="run_id"
          size="small"
          loading={loading}
          columns={columns}
          dataSource={runs}
          pagination={{ pageSize: 8, hideOnSinglePage: true }}
          locale={{ emptyText: <Empty description="还没有运行记录" /> }}
        />
      </Card>

      <Modal
        title={pendingDecision?.kind === 'approve' ? '审批通过' : '审批拒绝'}
        open={pendingDecision !== null}
        okText={pendingDecision?.kind === 'approve' ? '通过并继续归档' : '拒绝并退回'}
        okButtonProps={{ danger: pendingDecision?.kind === 'reject' }}
        cancelText="取消"
        confirmLoading={working !== ''}
        onCancel={() => setPendingDecision(null)}
        onOk={() => void decide()}
      >
        {pendingDecision && (
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <Alert
              type="warning"
              showIcon
              message={pendingDecision.run.approval?.title ?? '该运行需要人工确认'}
              description={pendingDecision.run.approval?.summary ?? ''}
            />
            <Input
              value={approver}
              onChange={(event) => setApprover(event.target.value)}
              placeholder="审批人（可选）"
              maxLength={64}
            />
            <Input.TextArea
              rows={3}
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              placeholder="审批意见（可选；拒绝时建议写明原因）"
              maxLength={500}
            />
          </Space>
        )}
      </Modal>

      <Modal
        title="运行详情"
        open={detail !== null}
        footer={null}
        width={880}
        onCancel={() => setDetail(null)}
      >
        {detail && <RunDetail run={detail} />}
      </Modal>
    </Space>
  )
}
