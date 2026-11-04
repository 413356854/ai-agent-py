import type { ReactNode } from 'react'
import { Card, Descriptions, Space, Table, Tag, Typography } from 'antd'
import type { TableProps } from 'antd'
import type { AgentFinding, ClaimData, ClaimInvoice } from '../types'

function toNumber(value: string | number | undefined): number {
  if (value === undefined || value === '') return 0
  const parsed = Number(String(value).replace(/,/g, ''))
  return Number.isFinite(parsed) ? parsed : 0
}

function money(value: string | number | undefined): string {
  return toNumber(value).toLocaleString('zh-CN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

interface ClaimDataCardProps {
  /** 单据原文：内置样例或用户上传的解析结果。 */
  claim: ClaimData
  /** 右侧标题，如样例编号 / 上传文件名。 */
  title?: ReactNode
  /** 左上角标签文本。 */
  badge?: string
  /** 预期结论（内置样例才有）。 */
  expected?: string
  /** 试跑结束后传入，命中的规则会标在对应发票行上。 */
  findings?: AgentFinding[]
  extra?: ReactNode
}

/** 「单据数据」卡片：把单据原文摊开，便于和审核结果逐项对照。 */
export function ClaimDataCard({ claim, title, badge, expected, findings = [], extra }: ClaimDataCardProps) {
  const hitsByInvoice = new Map<string, AgentFinding[]>()
  for (const finding of findings) {
    const number = typeof finding.evidence.number === 'string' ? finding.evidence.number : ''
    if (!number) continue
    hitsByInvoice.set(number, [...(hitsByInvoice.get(number) ?? []), finding])
  }

  const invoiceTotal = (claim.invoices ?? []).reduce((sum, invoice) => sum + toNumber(invoice.total), 0)
  const claimTotal = toNumber(claim.total)
  const diff = claimTotal - invoiceTotal
  const mismatch = Math.abs(diff) > 0.005

  const columns: TableProps<ClaimInvoice>['columns'] = [
    {
      title: '发票号',
      dataIndex: 'number',
      width: 150,
      render: (value: string) => (
        <Space size={4} wrap>
          <span>{value}</span>
          {(hitsByInvoice.get(value) ?? []).map((finding) => (
            <Tag key={finding.rule_id} color="red">
              {finding.rule_id}
            </Tag>
          ))}
        </Space>
      ),
    },
    { title: '开票日期', dataIndex: 'date', width: 108 },
    { title: '销售方', dataIndex: 'seller', ellipsis: true },
    {
      title: '票种',
      dataIndex: 'kind',
      width: 90,
      render: (value: string) => <Tag>{value}</Tag>,
    },
    {
      title: '金额',
      dataIndex: 'amount',
      width: 108,
      align: 'right',
      render: (value: string) => money(value),
    },
    {
      title: '税额',
      dataIndex: 'tax',
      width: 96,
      align: 'right',
      render: (value: string) => money(value),
    },
    {
      title: '价税合计',
      dataIndex: 'total',
      width: 116,
      align: 'right',
      render: (value: string) => <Typography.Text strong>{money(value)}</Typography.Text>,
    },
    {
      title: '明细 / 住宿',
      key: 'detail',
      width: 180,
      render: (_, invoice) => {
        const items = (invoice.items ?? []).map((item) => item.name).join('、')
        const stay = invoice.nights ? `${invoice.nights} 晚 · ${invoice.city_tier}` : ''
        return (
          <Typography.Text type="secondary">
            {[items, stay].filter(Boolean).join(' ｜ ') || '—'}
          </Typography.Text>
        )
      },
    },
  ]

  return (
    <Card
      size="small"
      title={
        <Space>
          <span>单据数据</span>
          {badge && <Tag color="blue">{badge}</Tag>}
          <Typography.Text type="secondary">{title}</Typography.Text>
        </Space>
      }
      extra={
        <Space>
          {expected && (
            <Typography.Text type="secondary">
              预期结论：
              <Tag color={expected === '通过' ? 'green' : expected === '退回' ? 'red' : 'gold'}>
                {expected}
              </Tag>
            </Typography.Text>
          )}
          {extra}
        </Space>
      }
    >
      <Descriptions column={3} size="small" bordered>
        <Descriptions.Item label="单据号">{claim.claim_id}</Descriptions.Item>
        <Descriptions.Item label="报销人">
          {claim.employee?.name || '—'}（{claim.employee?.dept || '—'}）
        </Descriptions.Item>
        <Descriptions.Item label="费用类型">{claim.category || '—'}</Descriptions.Item>
        <Descriptions.Item label="报销金额">￥{money(claim.total)}</Descriptions.Item>
        <Descriptions.Item label="提交日期">{claim.submit_date || '—'}</Descriptions.Item>
        <Descriptions.Item label="抽取置信度">
          {Math.round(toNumber(claim.ocr_confidence) * 100)}%
        </Descriptions.Item>
        <Descriptions.Item label="事由" span={3}>
          {claim.purpose || '—'}
        </Descriptions.Item>
        <Descriptions.Item label="审批链" span={3}>
          {(claim.approvals ?? []).map((item) => `${item.role} ${item.name}`).join(' › ') || '无'}
        </Descriptions.Item>
        <Descriptions.Item label="附件" span={3}>
          {(claim.attachments ?? []).join('、') || '无'}
        </Descriptions.Item>
      </Descriptions>

      <Table<ClaimInvoice>
        rowKey="number"
        size="small"
        style={{ marginTop: 16 }}
        pagination={false}
        columns={columns}
        dataSource={claim.invoices ?? []}
      />

      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 24, marginTop: 12 }}>
        <Typography.Text type="secondary">发票合计 ￥{money(invoiceTotal)}</Typography.Text>
        <Typography.Text type="secondary">报销金额 ￥{money(claimTotal)}</Typography.Text>
        <Typography.Text type={mismatch ? 'danger' : 'success'} strong>
          {mismatch ? `差额 ￥${money(diff)}` : '金额一致'}
        </Typography.Text>
      </div>
    </Card>
  )
}
