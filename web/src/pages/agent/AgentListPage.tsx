import { useCallback, useEffect, useState } from 'react'
import { ReloadOutlined, RobotOutlined } from '@ant-design/icons'
import { Alert, Button, Card, Col, Empty, Row, Space, Spin, Statistic, Tag, Typography } from 'antd'
import { useNavigate } from 'react-router-dom'
import { ApiError, api } from '../../api/client'
import type { AgentSummary } from '../../types'

function errorText(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return String(error)
}

/** Agent 应用列表：当前只有报销审核，后续在这里挂新应用。 */
export function AgentListPage() {
  const navigate = useNavigate()
  const [items, setItems] = useState<AgentSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      setItems(await api.agents())
      setError('')
    } catch (err) {
      setError(errorText(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load, reloadKey])

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Space style={{ width: '100%', justifyContent: 'space-between' }}>
        <div>
          <Typography.Title level={4} style={{ marginBottom: 4 }}>
            Agent 应用
          </Typography.Title>
          <Typography.Text type="secondary">
            每个应用是一张 LangGraph 图：节点即工序，边即流转，点进去可以看流程图并试跑。
          </Typography.Text>
        </div>
        <Button icon={<ReloadOutlined />} onClick={() => setReloadKey((value) => value + 1)}>
          刷新
        </Button>
      </Space>

      {error && <Alert type="error" showIcon message="加载失败" description={error} />}

      {loading ? (
        <div style={{ padding: 48, textAlign: 'center' }}>
          <Spin />
        </div>
      ) : items.length === 0 ? (
        <Empty description="还没有登记任何 Agent 应用" />
      ) : (
        <Row gutter={[16, 16]}>
          {items.map((item) => (
            <Col key={item.id} xs={24} lg={12} xl={8}>
              <Card
                hoverable
                onClick={() => navigate(`/agents/${item.id}`)}
                title={
                  <Space>
                    <RobotOutlined />
                    <span>{item.name}</span>
                    <Tag color="green">{item.status}</Tag>
                  </Space>
                }
                extra={<Typography.Link>查看流程图 →</Typography.Link>}
              >
                <Typography.Paragraph type="secondary" style={{ minHeight: 44 }}>
                  {item.summary}
                </Typography.Paragraph>
                <Space size={4} wrap>
                  {item.tags.map((tag) => (
                    <Tag key={tag}>{tag}</Tag>
                  ))}
                </Space>
                <div
                  style={{
                    display: 'flex',
                    gap: 32,
                    marginTop: 20,
                    paddingTop: 16,
                    borderTop: '1px solid rgba(128,128,128,0.2)',
                  }}
                >
                  <Statistic title="节点" value={item.node_count} />
                  <Statistic title="规则" value={item.rule_count} />
                  <Statistic title="版本" value={item.version} />
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      )}
    </Space>
  )
}
