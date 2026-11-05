import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloudUploadOutlined,
  CloseCircleOutlined,
  CopyOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  DownOutlined,
  DownloadOutlined,
  FileTextOutlined,
  InboxOutlined,
  ReloadOutlined,
  SplitCellsOutlined,
  SyncOutlined,
  UnorderedListOutlined,
} from '@ant-design/icons'
import {
  Alert,
  App,
  Button,
  Card,
  Col,
  Descriptions,
  Divider,
  Drawer,
  Dropdown,
  Empty,
  Form,
  Input,
  Popconfirm,
  Progress,
  Row,
  Select,
  Skeleton,
  Space,
  Statistic,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  Upload,
} from 'antd'
import type { MenuProps, TableProps } from 'antd'
import dayjs from 'dayjs'
import { useSearchParams } from 'react-router-dom'
import { ApiError, api } from '../../api/client'
import { MarkdownPreview } from '../../components/MarkdownPreview'
import { SplitPreviewModal } from '../../components/SplitPreviewModal'
import type {
  DocumentConfig,
  JobState,
  MarkdownOut,
  ParseJob,
  UploadOptions,
} from '../../types'

const STATE_META: Record<JobState, { color: string; icon: ReactNode }> = {
  queued: { color: 'default', icon: <ClockCircleOutlined /> },
  uploading: { color: 'processing', icon: <CloudUploadOutlined /> },
  parsing: { color: 'processing', icon: <SyncOutlined spin /> },
  downloading: { color: 'cyan', icon: <DownloadOutlined /> },
  done: { color: 'success', icon: <CheckCircleOutlined /> },
  failed: { color: 'error', icon: <CloseCircleOutlined /> },
}

const DEFAULT_OPTIONS: UploadOptions = {
  model_version: 'vlm',
  language: 'ch',
  enable_formula: true,
  enable_table: true,
  is_ocr: false,
  page_ranges: '',
}

interface PreviewState {
  open: boolean
  job: ParseJob | null
  data: MarkdownOut | null
  loading: boolean
  error: string
}

type PendingFileStatus = 'queued' | 'submitting' | 'failed'

interface PendingFile {
  uid: string
  file: File
  status: PendingFileStatus
  error: string
}

let pendingFileSeed = 0

function nextPendingUid(): string {
  pendingFileSeed += 1
  return `pending-${Date.now()}-${pendingFileSeed}`
}

function errorText(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return String(error)
}

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)))
  const value = bytes / 1024 ** index
  return `${index === 0 ? value : value.toFixed(1)} ${units[index]}`
}

export function DocumentParsePage() {
  const { message } = App.useApp()
  const [searchParams, setSearchParams] = useSearchParams()
  const deepLinkHandled = useRef(false)

  const [config, setConfig] = useState<DocumentConfig | null>(null)
  const [configError, setConfigError] = useState('')
  const [jobs, setJobs] = useState<ParseJob[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([])
  const [options, setOptions] = useState<UploadOptions>(DEFAULT_OPTIONS)
  const [uploading, setUploading] = useState(false)
  const [splitJob, setSplitJob] = useState<ParseJob | null>(null)
  const [preview, setPreview] = useState<PreviewState>({
    open: false,
    job: null,
    data: null,
    loading: false,
    error: '',
  })

  const refreshJobs = useCallback(
    async (silent = false) => {
      if (!silent) setRefreshing(true)
      try {
        setJobs(await api.list())
      } catch (error) {
        if (!silent) message.error(errorText(error))
      } finally {
        if (!silent) setRefreshing(false)
      }
    },
    [message],
  )

  useEffect(() => {
    let cancelled = false
    async function bootstrap() {
      setLoading(true)
      try {
        const cfg = await api.config()
        if (cancelled) return
        setConfig(cfg)
        setOptions({
          model_version: cfg.default_options.model_version,
          language: cfg.default_options.language,
          enable_formula: cfg.default_options.enable_formula,
          enable_table: cfg.default_options.enable_table,
          is_ocr: cfg.default_options.is_ocr,
          page_ranges: '',
        })
      } catch (error) {
        if (!cancelled) setConfigError(errorText(error))
      }
      try {
        const list = await api.list()
        if (!cancelled) setJobs(list)
      } catch (error) {
        if (!cancelled) message.error(errorText(error))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void bootstrap()
    return () => {
      cancelled = true
    }
  }, [message])

  const hasActiveJobs = useMemo(() => jobs.some((job) => !job.is_terminal), [jobs])

  useEffect(() => {
    if (!hasActiveJobs) return
    const timer = window.setInterval(() => {
      void refreshJobs(true)
    }, 2500)
    return () => window.clearInterval(timer)
  }, [hasActiveJobs, refreshJobs])

  const stats = useMemo(
    () => ({
      total: jobs.length,
      running: jobs.filter((job) => !job.is_terminal).length,
      done: jobs.filter((job) => job.state === 'done').length,
      ingested: jobs.filter((job) => job.rag_ingested).length,
      failed: jobs.filter((job) => job.state === 'failed').length,
    }),
    [jobs],
  )

  const pendingBytes = pendingFiles.reduce((total, item) => total + item.file.size, 0)

  const beforeUpload = (candidate: File) => {
    if (uploading) return Upload.LIST_IGNORE
    const suffix = `.${candidate.name.split('.').pop()?.toLowerCase() ?? ''}`
    if (config && !config.accepted_suffixes.includes(suffix)) {
      message.error(`不支持的文件类型：${suffix || '未知'}`)
      return Upload.LIST_IGNORE
    }
    if (config && candidate.size > config.max_file_mb * 1024 * 1024) {
      message.error(`文件超过 ${config.max_file_mb} MB 上限`)
      return Upload.LIST_IGNORE
    }
    setPendingFiles((previous) => [
      ...previous,
      { uid: nextPendingUid(), file: candidate, status: 'queued', error: '' },
    ])
    return Upload.LIST_IGNORE
  }

  const handleUpload = async () => {
    if (!pendingFiles.length) {
      message.warning('请先选择要解析的文件')
      return
    }
    if (!config) {
      message.warning('解析配置尚未加载完成')
      return
    }

    // 一次点击按队列顺序逐个提交；成功项移出队列，失败项保留以便重试。
    const batch = [...pendingFiles]
    const chosenOptions = { ...options }
    setUploading(true)
    let submitted = 0
    let failed = 0

    try {
      for (const item of batch) {
        setPendingFiles((previous) =>
          previous.map((pending) =>
            pending.uid === item.uid
              ? { ...pending, status: 'submitting', error: '' }
              : pending,
          ),
        )
        try {
          const job = await api.parse(item.file, chosenOptions)
          submitted += 1
          setJobs((previous) => [job, ...previous.filter((existing) => existing.id !== job.id)])
          setPendingFiles((previous) => previous.filter((pending) => pending.uid !== item.uid))
        } catch (error) {
          failed += 1
          const reason = errorText(error)
          setPendingFiles((previous) =>
            previous.map((pending) =>
              pending.uid === item.uid
                ? { ...pending, status: 'failed', error: reason }
                : pending,
            ),
          )
        }
      }
    } finally {
      setUploading(false)
    }

    if (submitted > 0) {
      void refreshJobs(true)
    }
    if (failed === 0) {
      message.success(`已提交 ${submitted} 个解析任务`)
    } else if (submitted === 0) {
      message.error(`${failed} 个文件提交失败，已保留在队列中`)
    } else {
      message.warning(`已提交 ${submitted} 个，${failed} 个失败；失败文件保留在队列中`)
    }
  }

  const openPreview = async (job: ParseJob) => {
    setSearchParams({ job: job.id }, { replace: true })
    setPreview({ open: true, job, data: null, loading: true, error: '' })
    try {
      const data = await api.markdown(job.id)
      setPreview({ open: true, job, data, loading: false, error: '' })
    } catch (error) {
      setPreview({ open: true, job, data: null, loading: false, error: errorText(error) })
    }
  }

  const closePreview = () => {
    setPreview((previous) => ({ ...previous, open: false }))
    setSearchParams({}, { replace: true })
  }

  // 支持 `?job=<id>` 直接打开某份文档的预览（链接可分享）
  const deepLinkJobId = searchParams.get('job')
  useEffect(() => {
    if (deepLinkHandled.current || !deepLinkJobId) return
    const target = jobs.find((item) => item.id === deepLinkJobId)
    if (!target) return
    deepLinkHandled.current = true
    void openPreview(target)
    // openPreview 每次渲染都会重建，这里只关心 job 参数与列表
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deepLinkJobId, jobs])

  const handleDelete = async (job: ParseJob) => {
    try {
      await api.remove(job.id)
      message.success(`已删除：${job.file_name}`)
      setJobs((previous) => previous.filter((item) => item.id !== job.id))
    } catch (error) {
      message.error(errorText(error))
    }
  }

  const download = (jobId: string, kind: 'markdown' | 'zip' | 'source') => {
    const anchor = document.createElement('a')
    anchor.href = api.downloadUrl(jobId, kind)
    anchor.rel = 'noopener'
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
  }

  const handleCopy = async () => {
    if (!preview.data) return
    try {
      await navigator.clipboard.writeText(preview.data.markdown)
      message.success('Markdown 已复制到剪贴板')
    } catch {
      message.error('复制失败，请手动选择文本复制')
    }
  }

  const downloadItems = (job: ParseJob): MenuProps['items'] => {
    const items: NonNullable<MenuProps['items']> = [
      { key: 'markdown', label: 'Markdown（.md）', onClick: () => download(job.id, 'markdown') },
    ]
    if (job.zip_file) {
      items.push({
        key: 'zip',
        label: '解析结果压缩包（.zip）',
        onClick: () => download(job.id, 'zip'),
      })
    }
    items.push(
      { type: 'divider' },
      { key: 'source', label: '原始文件', onClick: () => download(job.id, 'source') },
    )
    return items
  }

  const columns: TableProps<ParseJob>['columns'] = [
    {
      title: '文件名',
      key: 'file_name',
      width: 240,
      render: (_, job) => (
        <Space direction="vertical" size={0} style={{ maxWidth: 230 }}>
          <Typography.Text strong ellipsis={{ tooltip: job.file_name }}>
            {job.file_name}
          </Typography.Text>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {job.suffix.toUpperCase() || '—'} · {formatBytes(job.file_size)}
            {job.markdown_chars ? ` · ${job.markdown_chars.toLocaleString()} 字符` : ''}
            {job.rag_ingest ? ` · RAG 已入库 ${job.rag_ingest.chunks} 片段` : ''}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: '解析配置',
      key: 'options',
      width: 140,
      render: (_, job) => (
        <Space size={4} wrap>
          <Tag color="blue">{job.options.model_version}</Tag>
          <Tag>{job.options.language}</Tag>
        </Space>
      ),
    },
    {
      title: '状态',
      key: 'state',
      width: 180,
      render: (_, job) => {
        const meta = STATE_META[job.state] ?? STATE_META.queued
        return (
          <Space direction="vertical" size={2} style={{ width: '100%' }}>
            <Space size={4} wrap>
              <Tag color={meta.color} icon={meta.icon} style={{ marginInlineEnd: 0 }}>
                {job.state_label}
              </Tag>
              {job.rag_ingested && (
                <Tooltip
                  title={`已写入 RAG 表 ${job.rag_ingest?.collection ?? ''}（${
                    job.rag_ingest?.chunks ?? 0
                  } 个切片）；重新入库会先删除旧切片`}
                >
                  <Tag
                    color="purple"
                    icon={<DatabaseOutlined />}
                    style={{ marginInlineEnd: 0 }}
                  >
                    已入库
                  </Tag>
                </Tooltip>
              )}
            </Space>
            {!job.is_terminal && (
              <Progress
                percent={Math.round(job.progress * 100)}
                size="small"
                showInfo={false}
                status="active"
              />
            )}
            {job.state === 'failed' && (
              <Tooltip title={job.err_msg}>
                <Typography.Text type="danger" style={{ fontSize: 12 }} ellipsis>
                  {job.err_msg}
                </Typography.Text>
              </Tooltip>
            )}
          </Space>
        )
      },
    },
    {
      title: '页数',
      key: 'pages',
      width: 76,
      align: 'center',
      render: (_, job) =>
        job.total_pages ? `${job.extracted_pages}/${job.total_pages}` : '—',
    },
    {
      title: '耗时',
      key: 'elapsed',
      width: 76,
      render: (_, job) =>
        job.elapsed_seconds ? `${job.elapsed_seconds.toFixed(1)}s` : '—',
    },
    {
      title: '提交时间',
      key: 'created_at',
      width: 160,
      render: (_, job) => (
        <Typography.Text type="secondary">
          {dayjs(job.created_at).format('YYYY-MM-DD HH:mm:ss')}
        </Typography.Text>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 260,
      fixed: 'right',
      render: (_, job) => (
        <Space size={0} wrap>
          <Button
            type="link"
            size="small"
            disabled={!job.markdown_file}
            onClick={() => void openPreview(job)}
          >
            预览
          </Button>
          <Button
            type="link"
            size="small"
            icon={<SplitCellsOutlined />}
            disabled={job.state !== 'done' || !job.markdown_file}
            onClick={() => setSplitJob(job)}
          >
            切分
          </Button>
          <Dropdown menu={{ items: downloadItems(job) }} disabled={!job.markdown_file}>
            <Button type="link" size="small" disabled={!job.markdown_file}>
              下载 <DownOutlined />
            </Button>
          </Dropdown>
          <Popconfirm
            title="删除该任务？"
            description="会同时删除解析结果与源文件，不可恢复。"
            okText="删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
            onConfirm={() => void handleDelete(job)}
          >
            <Button type="link" size="small" danger>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <Typography.Title level={4} style={{ margin: 0 }}>
            文档解析
          </Typography.Title>
          <Typography.Text type="secondary">
            上传文档 → MinerU 精准解析 API → Markdown / 结构化结果 → 写入知识库
          </Typography.Text>
        </div>
        <Space>
          <Button
            icon={<ReloadOutlined />}
            loading={refreshing}
            onClick={() => void refreshJobs()}
          >
            刷新
          </Button>
          <Button icon={<FileTextOutlined />} href="/docs" target="_blank">
            API 文档
          </Button>
        </Space>
      </div>

      {configError && (
        <Alert
          type="error"
          showIcon
          message="无法读取解析配置"
          description={configError}
          style={{ marginBottom: 16 }}
        />
      )}

      {config && !config.configured && (
        <Alert
          type="warning"
          showIcon
          message="尚未配置 MinerU Token"
          description="在 .env 中设置 MINERU_API_KEY 后重启服务。Token 申请：https://mineru.net/apiManage/token"
          style={{ marginBottom: 16 }}
        />
      )}

      {config && !config.rag_configured && (
        <Alert
          type="warning"
          showIcon
          message="尚未配置 Embedding API Key"
          description="切分预览不受影响，但「入库 RAG」需要 DASHSCOPE_API_KEY（百炼 text-embedding-v4）。"
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={[16, 16]} className="stat-row">
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic title="解析任务" value={stats.total} />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic title="进行中" value={stats.running} valueStyle={{ color: '#1677ff' }} />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic title="已完成" value={stats.done} valueStyle={{ color: '#3f8600' }} />
            {stats.ingested > 0 && (
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                其中 RAG 已入库 {stats.ingested}
              </Typography.Text>
            )}
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic
              title="失败"
              value={stats.failed}
              valueStyle={{ color: stats.failed ? '#cf1322' : undefined }}
            />
          </Card>
        </Col>
      </Row>

      <Card
        className="panel"
        title={
          <Space>
            <CloudUploadOutlined />
            上传文档
          </Space>
        }
        extra={
          config && (
            <Space size={6}>
              <Tag color="geekblue">默认 {config.model_version}</Tag>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                可批量 · 单文件 ≤ {config.max_file_mb} MB
              </Typography.Text>
            </Space>
          )
        }
      >
        <Row gutter={24}>
          <Col xs={24} lg={10}>
            <div className="upload-dragger-wrap">
              <Upload.Dragger
                accept={config?.accepted_suffixes.join(',')}
                multiple
                disabled={uploading}
                fileList={[]}
                beforeUpload={beforeUpload}
                showUploadList={false}
              >
                <p className="ant-upload-drag-icon">
                  <InboxOutlined />
                </p>
                <p className="ant-upload-text">点击或拖拽多个文件到此区域</p>
                <p className="ant-upload-hint">
                  可分批反复添加，文件会累积到下方队列；支持 PDF / Word / PPT / 图片，以及可直接切分的 Markdown（.md）与 JSON
                </p>
              </Upload.Dragger>
            </div>

            <div className="upload-queue">
              <div className="upload-queue-head">
                <span className="upload-queue-title">
                  待解析队列
                  <Tag style={{ marginInlineEnd: 0 }}>{pendingFiles.length}</Tag>
                </span>
                <Space size={4}>
                  {pendingFiles.length > 0 && (
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      {formatBytes(pendingBytes)}
                    </Typography.Text>
                  )}
                  <Button
                    type="link"
                    size="small"
                    disabled={uploading || pendingFiles.length === 0}
                    onClick={() => setPendingFiles([])}
                  >
                    清空
                  </Button>
                </Space>
              </div>
              {pendingFiles.length === 0 ? (
                <div className="upload-queue-empty">暂未添加文件，可分批拖入，队列会自动累积</div>
              ) : (
                <div className="upload-queue-list">
                  {pendingFiles.map((item) => (
                    <div className="upload-queue-item" key={item.uid}>
                      <FileTextOutlined className="upload-queue-icon" />
                      <div className="upload-queue-meta">
                        <Typography.Text
                          ellipsis={{ tooltip: item.file.name }}
                          style={{ fontSize: 13 }}
                        >
                          {item.file.name}
                        </Typography.Text>
                        <span className="upload-queue-size">{formatBytes(item.file.size)}</span>
                      </div>
                      {item.status === 'submitting' && (
                        <Tag
                          color="processing"
                          icon={<SyncOutlined spin />}
                          style={{ marginInlineEnd: 0 }}
                        >
                          提交中
                        </Tag>
                      )}
                      {item.status === 'failed' && (
                        <Tooltip title={item.error}>
                          <Tag
                            color="error"
                            icon={<CloseCircleOutlined />}
                            style={{ marginInlineEnd: 0 }}
                          >
                            失败
                          </Tag>
                        </Tooltip>
                      )}
                      {item.status === 'queued' && (
                        <Tag style={{ marginInlineEnd: 0 }}>待提交</Tag>
                      )}
                      <Tooltip title="移出队列">
                        <Button
                          type="text"
                          size="small"
                          danger
                          className="upload-queue-remove"
                          aria-label={`移出队列：${item.file.name}`}
                          disabled={uploading}
                          icon={<DeleteOutlined />}
                          onClick={() =>
                            setPendingFiles((previous) =>
                              previous.filter((pending) => pending.uid !== item.uid),
                            )
                          }
                        />
                      </Tooltip>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Col>
          <Col xs={24} lg={14}>
            <Form layout="vertical" size="middle" className="parse-form">
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="模型版本"
                    tooltip="vlm：精度最高（推荐）；pipeline：速度更快；HTML 文件需选 MinerU-HTML"
                  >
                    <Select
                      value={options.model_version}
                      disabled={uploading}
                      options={(config?.model_versions ?? ['vlm', 'pipeline']).map(
                        (value) => ({ value, label: value }),
                      )}
                      onChange={(value) =>
                        setOptions((previous) => ({ ...previous, model_version: value }))
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="文档语言" tooltip="OCR 识别语言，扫描件或图片务必选择正确">
                    <Select
                      showSearch
                      value={options.language}
                      disabled={uploading}
                      options={(config?.languages ?? ['ch']).map((value) => ({
                        value,
                        label: value,
                      }))}
                      onChange={(value) =>
                        setOptions((previous) => ({ ...previous, language: value }))
                      }
                    />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item label="公式识别">
                    <Switch
                      checked={options.enable_formula}
                      disabled={uploading}
                      onChange={(checked) =>
                        setOptions((previous) => ({ ...previous, enable_formula: checked }))
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="表格识别">
                    <Switch
                      checked={options.enable_table}
                      disabled={uploading}
                      onChange={(checked) =>
                        setOptions((previous) => ({ ...previous, enable_table: checked }))
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="扫描件 OCR" tooltip="纯图片 / 扫描版 PDF 需要开启">
                    <Switch
                      checked={options.is_ocr}
                      disabled={uploading}
                      onChange={(checked) =>
                        setOptions((previous) => ({ ...previous, is_ocr: checked }))
                      }
                    />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="页码范围" tooltip='留空解析全部；格式如 "1-10" 或 "2,4-6"'>
                    <Input
                      allowClear
                      placeholder="例如 1-10（留空 = 全部）"
                      value={options.page_ranges}
                      disabled={uploading}
                      onChange={(event) =>
                        setOptions((previous) => ({
                          ...previous,
                          page_ranges: event.target.value,
                        }))
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label=" ">
                    <Space>
                      <Button
                        type="primary"
                        icon={<CloudUploadOutlined />}
                        loading={uploading}
                        disabled={!pendingFiles.length || !config}
                        onClick={() => void handleUpload()}
                      >
                        {uploading ? '逐个提交中…' : '开始解析'}
                      </Button>
                      <Button
                        disabled={uploading}
                        onClick={() => {
                          setPendingFiles([])
                          setOptions(
                            config ? defaultOptionsFrom(config) : { ...DEFAULT_OPTIONS },
                          )
                        }}
                      >
                        重置
                      </Button>
                    </Space>
                  </Form.Item>
                </Col>
              </Row>
            </Form>
          </Col>
        </Row>
      </Card>

      <Card
        className="panel"
        title={
          <Space>
            <UnorderedListOutlined />
            解析任务
          </Space>
        }
        extra={
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            共 {jobs.length} 条{hasActiveJobs ? ' · 正在自动刷新' : ''}
          </Typography.Text>
        }
      >
        <Table<ParseJob>
          rowKey="id"
          size="middle"
          loading={loading}
          columns={columns}
          dataSource={jobs}
          scroll={{ x: 1080 }}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 条`,
          }}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="还没有解析任务，先添加文档试试"
              />
            ),
          }}
        />
        {config && (
          <div className="meta-line">
            MinerU 接口：{config.base_url} · 结果目录：{config.output_dir} · 支持{' '}
            {config.accepted_suffixes.length} 种文件类型
          </div>
        )}
      </Card>

      <Drawer
        open={preview.open}
        onClose={closePreview}
        width={880}
        title={preview.job ? `Markdown 预览 · ${preview.job.file_name}` : 'Markdown 预览'}
        extra={
          <Space>
            <Button
              size="small"
              icon={<CopyOutlined />}
              disabled={!preview.data}
              onClick={() => void handleCopy()}
            >
              复制
            </Button>
            <Button
              size="small"
              type="primary"
              icon={<DownloadOutlined />}
              disabled={!preview.data}
              onClick={() => preview.job && download(preview.job.id, 'markdown')}
            >
              下载 Markdown
            </Button>
          </Space>
        }
      >
        {preview.loading && <Skeleton active paragraph={{ rows: 12 }} />}
        {!preview.loading && preview.error && (
          <Alert type="error" showIcon message="读取解析结果失败" description={preview.error} />
        )}
        {!preview.loading && preview.data && (
          <>
            <Descriptions
              size="small"
              column={3}
              items={[
                {
                  key: 'chars',
                  label: '字符数',
                  children: preview.data.characters.toLocaleString(),
                },
                { key: 'images', label: '图片', children: preview.data.assets },
                {
                  key: 'model',
                  label: '模型',
                  children: preview.job?.options.model_version ?? '—',
                },
              ]}
            />
            <Divider style={{ margin: '12px 0 16px' }} />
            <MarkdownPreview
              markdown={preview.data.markdown}
              assetsBase={preview.data.assets_base}
            />
          </>
        )}
      </Drawer>

      <SplitPreviewModal
        open={Boolean(splitJob)}
        job={splitJob}
        onClose={() => setSplitJob(null)}
        onIngested={() => {
          void refreshJobs(true)
        }}
      />
    </div>
  )
}

function defaultOptionsFrom(config: DocumentConfig): UploadOptions {
  return {
    model_version: config.default_options.model_version,
    language: config.default_options.language,
    enable_formula: config.default_options.enable_formula,
    enable_table: config.default_options.enable_table,
    is_ocr: config.default_options.is_ocr,
    page_ranges: '',
  }
}
