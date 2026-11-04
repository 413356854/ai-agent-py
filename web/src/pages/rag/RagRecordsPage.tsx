import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  CopyOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  MessageOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import {
  Alert,
  App,
  Button,
  Card,
  Col,
  Descriptions,
  Divider,
  Empty,
  Input,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Tooltip,
  Typography,
} from 'antd'
import type { TableProps } from 'antd'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ApiError, api } from '../../api/client'
import type { RagChunkList, RagChunkRecord, RagOverview } from '../../types'

const PAGE_SIZE_OPTIONS = [10, 20, 50, 100]

function errorText(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return String(error)
}

function shortChunkId(chunkId: string): string {
  return chunkId.replace(/^chunk-/, '').slice(0, 12)
}

/** RAG 记录：分页浏览 Milvus `rag_chunk`，按文档 / 正文关键词过滤。 */
export function RagRecordsPage() {
  const { message } = App.useApp()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const docId = searchParams.get('doc_id') ?? ''

  const [overview, setOverview] = useState<RagOverview | null>(null)
  const [overviewError, setOverviewError] = useState('')
  const [keywordInput, setKeywordInput] = useState('')
  const [keyword, setKeyword] = useState('')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [data, setData] = useState<RagChunkList | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [reloadKey, setReloadKey] = useState(0)
  const [preview, setPreview] = useState<RagChunkRecord | null>(null)

  const loadOverview = useCallback(async () => {
    try {
      const next = await api.ragOverview()
      setOverview(next)
      setOverviewError('')
    } catch (error) {
      setOverviewError(errorText(error))
    }
  }, [])

  useEffect(() => {
    void loadOverview()
  }, [loadOverview])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    api
      .ragChunks({
        doc_id: docId || undefined,
        keyword: keyword || undefined,
        page,
        page_size: pageSize,
      })
      .then((next) => {
        if (!cancelled) setData(next)
      })
      .catch((error) => {
        if (!cancelled) message.error(errorText(error))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [docId, keyword, page, pageSize, reloadKey, message])

  const docOptions = useMemo(() => {
    const options = (overview?.docs ?? []).map((doc) => ({
      value: doc.doc_id,
      label: `${doc.file_name ?? doc.doc_id}（${doc.chunks} 块）`,
    }))
    if (docId && !options.some((option) => option.value === docId)) {
      // 深链进来但文档分布里没有（MySQL 不可读 / 超过扫描上限）时也要能显示当前值
      options.unshift({ value: docId, label: docId })
    }
    return options
  }, [overview, docId])

  const stats = useMemo(
    () => ({
      chunks: overview?.total_chunks ?? null,
      docs: overview?.total_docs ?? null,
      filtered: data?.total ?? null,
      dim: overview?.dim ?? null,
    }),
    [overview, data],
  )

  const refresh = async () => {
    setRefreshing(true)
    try {
      await loadOverview()
      setReloadKey((key) => key + 1)
    } finally {
      setRefreshing(false)
    }
  }

  const selectDoc = (value: string | undefined) => {
    setPage(1)
    if (value) {
      setSearchParams({ doc_id: value }, { replace: true })
    } else {
      setSearchParams({}, { replace: true })
    }
  }

  const applyKeyword = (value: string) => {
    setKeyword(value.trim())
    setPage(1)
    setReloadKey((key) => key + 1)
  }

  const resetFilters = () => {
    setKeywordInput('')
    setKeyword('')
    setPage(1)
    setSearchParams({}, { replace: true })
    setReloadKey((key) => key + 1)
  }

  const copyContent = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      message.success('块正文已复制')
    } catch {
      message.error('复制失败，请打开片段详情后手动选择文本')
    }
  }

  const columns: TableProps<RagChunkRecord>['columns'] = [
    {
      title: '块 ID',
      dataIndex: 'chunk_id',
      width: 168,
      render: (chunkId: string) => (
        <Space size={2}>
          <Tooltip title={chunkId}>
            <Typography.Text className="mono-text">
              {shortChunkId(chunkId)}
            </Typography.Text>
          </Tooltip>
          <Typography.Text
            className="mono-text"
            copyable={{ text: chunkId, tooltips: ['复制块 ID', '已复制'] }}
          />
        </Space>
      ),
    },
    {
      title: '文档',
      dataIndex: 'doc_name',
      width: 260,
      render: (_value: string | null, record) => (
        <Space orientation="vertical" size={0} style={{ maxWidth: 240 }}>
          <Typography.Text strong ellipsis={{ tooltip: record.doc_name ?? record.doc_id }}>
            {record.doc_name ?? '未知文档'}
          </Typography.Text>
          <Typography.Text
            type="secondary"
            className="mono-text"
            ellipsis={{ tooltip: record.doc_id }}
            style={{ maxWidth: 240 }}
          >
            {shortChunkId(record.doc_id)}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: '块序号',
      dataIndex: 'chunk_index',
      width: 84,
      align: 'center',
      render: (index: number) => `#${index + 1}`,
    },
    {
      title: '页码',
      dataIndex: 'page',
      width: 72,
      align: 'center',
      render: (pageNumber: number | null) => pageNumber ?? '—',
    },
    {
      title: (
        <span>
          内容{' '}
          <Typography.Text type="secondary" style={{ fontSize: 11, fontWeight: 'normal' }}>
            点击查看全文
          </Typography.Text>
        </span>
      ),
      dataIndex: 'content',
      render: (_, record) => (
        <div
          className="record-content-click"
          role="button"
          tabIndex={0}
          aria-label={`查看完整片段：${record.doc_name ?? record.doc_id} 第 ${record.chunk_index + 1} 块`}
          onClick={() => setPreview(record)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              setPreview(record)
            }
          }}
        >
          <Typography.Paragraph className="record-content" ellipsis={{ rows: 2 }}>
            {record.content}
          </Typography.Paragraph>
        </div>
      ),
    },
    {
      title: '字数',
      dataIndex: 'characters',
      width: 84,
      align: 'right',
      render: (characters: number) => characters.toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      width: 132,
      fixed: 'right',
      render: (_, record) => (
        <Space size={0}>
          <Tooltip title="在「文档解析」里打开原文">
            <Button
              type="link"
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => navigate(`/rag/documents?job=${record.doc_id}`)}
            >
              原文
            </Button>
          </Tooltip>
          <Tooltip title="复制块正文">
            <Button
              type="link"
              size="small"
              icon={<CopyOutlined />}
              onClick={() => void copyContent(record.content)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const hasFilters = Boolean(docId || keyword)

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <Typography.Title level={4} style={{ margin: 0 }}>
            RAG 记录
          </Typography.Title>
          <Typography.Text type="secondary">
            Milvus <code>rag_chunk</code> 里的检索块：按文档 / 关键词查询，核对接入知识库的内容
          </Typography.Text>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} loading={refreshing} onClick={() => void refresh()}>
            刷新
          </Button>
          <Button icon={<MessageOutlined />} onClick={() => navigate('/rag/chat')}>
            去问答
          </Button>
        </Space>
      </div>

      {overviewError && (
        <Alert
          type="error"
          showIcon
          message="无法读取 RAG 集合状态"
          description={overviewError}
        />
      )}
      {overview && !overview.ready && (
        <Alert
          type="error"
          showIcon
          message="RAG 集合不可用"
          description={
            overview.error ??
            `请确认 Milvus（${overview.milvus_uri}）已启动，并已创建 ${overview.database}.${overview.collection}`
          }
        />
      )}
      {overview?.ready && !overview.doc_names_available && (
        <Alert
          type="info"
          showIcon
          message="没有读到 MySQL 任务表"
          description="文档列只显示 doc_id；配置 MYSQL_URL 并建好 tb_document_parse_job 后即可显示文件名。"
        />
      )}
      {data?.truncated && (
        <Alert
          type="info"
          showIcon
          message="该文档的块数超过单次扫描上限（16384）"
          description="分页只覆盖已有块的一部分；可以按关键词缩小范围后查看。"
        />
      )}

      <Row gutter={[16, 16]} className="stat-row">
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic title="检索块" value={stats.chunks ?? '—'} />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic title="文档" value={stats.docs ?? '—'} />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic title="当前筛选结果" value={stats.filtered ?? '—'} />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic title="向量维度" value={stats.dim ?? '—'} suffix={stats.dim ? '维' : ''} />
          </Card>
        </Col>
      </Row>

      <Card
        className="panel"
        title={
          <Space>
            <DatabaseOutlined />
            记录查询
          </Space>
        }
        extra={
          overview?.ready ? (
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              {overview.database}.{overview.collection} · {overview.metric} ·{' '}
              {overview.analyzer} 分词
            </Typography.Text>
          ) : null
        }
      >
        <Space size={12} wrap className="record-filters">
          <Select
            showSearch
            allowClear
            optionFilterProp="label"
            placeholder="全部文档"
            value={docId || undefined}
            options={docOptions}
            style={{ minWidth: 320, maxWidth: 520 }}
            onChange={(value) => selectDoc(value)}
          />
          <Input.Search
            allowClear
            enterButton="搜索"
            placeholder="正文关键词（中缀匹配，留空 = 全部）"
            value={keywordInput}
            style={{ width: 360 }}
            onChange={(event) => setKeywordInput(event.target.value)}
            onSearch={(value) => applyKeyword(value)}
          />
          {keyword && (
            <Tag
              color="blue"
              closable
              onClose={() => {
                setKeywordInput('')
                applyKeyword('')
              }}
            >
              关键词：{keyword}
            </Tag>
          )}
          {hasFilters && (
            <Button type="link" onClick={resetFilters}>
              清空筛选
            </Button>
          )}
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            共 {data?.total ?? 0} 条
          </Typography.Text>
        </Space>

        <Table<RagChunkRecord>
          rowKey="chunk_id"
          size="middle"
          style={{ marginTop: 16 }}
          loading={loading}
          columns={columns}
          dataSource={data?.items ?? []}
          scroll={{ x: 1120 }}
          pagination={{
            current: page,
            pageSize,
            total: data?.total ?? 0,
            showSizeChanger: true,
            pageSizeOptions: PAGE_SIZE_OPTIONS,
            showTotal: (total) => `共 ${total} 条`,
            onChange: (nextPage, nextPageSize) => {
              if (nextPageSize !== pageSize) {
                setPage(1)
                setPageSize(nextPageSize)
                return
              }
              setPage(nextPage)
            },
          }}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={
                  hasFilters
                    ? '没有匹配的检索块，换个关键词或清空筛选试试'
                    : '知识库还没有切片，先到「文档解析」切分后点「入库 RAG」'
                }
              />
            ),
          }}
        />

        <div className="meta-line">
          集合 <code>{overview?.collection ?? '—'}</code> · 向量维度 {stats.dim ?? '—'} · 度量{' '}
          {overview?.metric ?? '—'} · 分页由 Milvus <code>count(*)</code> 聚合给出总数
        </div>
      </Card>

      <Modal
        open={Boolean(preview)}
        onCancel={() => setPreview(null)}
        width={880}
        destroyOnHidden
        title={
          preview ? (
            <Space>
              <FileTextOutlined />
              片段详情 · {preview.doc_name ?? preview.doc_id} · 第 {preview.chunk_index + 1} 块
            </Space>
          ) : (
            '片段详情'
          )
        }
        footer={
          <Space>
            <Button onClick={() => setPreview(null)}>关闭</Button>
            <Button
              type="primary"
              icon={<CopyOutlined />}
              disabled={!preview}
              onClick={() => preview && void copyContent(preview.content)}
            >
              复制全文
            </Button>
          </Space>
        }
      >
        {preview && (
          <>
            <Descriptions
              size="small"
              column={2}
              items={[
                {
                  key: 'doc',
                  label: '文档',
                  span: 2,
                  children: preview.doc_name ?? '未知文档',
                },
                {
                  key: 'doc_id',
                  label: '文档 ID',
                  span: 2,
                  children: (
                    <Typography.Text className="mono-text" copyable={{ text: preview.doc_id }}>
                      {preview.doc_id}
                    </Typography.Text>
                  ),
                },
                {
                  key: 'chunk_id',
                  label: '块 ID',
                  span: 2,
                  children: (
                    <Typography.Text className="mono-text" copyable={{ text: preview.chunk_id }}>
                      {preview.chunk_id}
                    </Typography.Text>
                  ),
                },
                { key: 'index', label: '块序号', children: `#${preview.chunk_index + 1}` },
                { key: 'page', label: '页码', children: preview.page ?? '—' },
                {
                  key: 'chars',
                  label: '字数',
                  children: preview.characters.toLocaleString(),
                },
              ]}
            />
            <Divider style={{ margin: '12px 0' }} />
            <pre className="record-full-content">{preview.content}</pre>
          </>
        )}
      </Modal>
    </div>
  )
}
