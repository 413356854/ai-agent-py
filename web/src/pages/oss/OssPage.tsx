import { useCallback, useEffect, useState } from 'react'
import {
  CloudServerOutlined,
  CloudUploadOutlined,
  CopyOutlined,
  DeleteOutlined,
  DownloadOutlined,
  LinkOutlined,
  ReloadOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import {
  Alert,
  App,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Image,
  Input,
  Popconfirm,
  Row,
  Space,
  Statistic,
  Table,
  Tag,
  Tooltip,
  Typography,
  Upload,
} from 'antd'
import type { TableProps, UploadProps } from 'antd'
import dayjs from 'dayjs'
import { ApiError, api } from '../../api/client'
import type { OssImage, OssObject, OssOverview } from '../../types'

interface PreviewImage {
  key: string
  url: string
  downloadUrl: string
  expiresIn: number
}

function errorText(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return String(error)
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(2)} MB`
}

function formatExpiry(seconds: number): string {
  if (seconds % 86400 === 0) return `${seconds / 86400} 天`
  if (seconds % 3600 === 0) return `${seconds / 3600} 小时`
  if (seconds % 60 === 0) return `${seconds / 60} 分钟`
  return `${seconds} 秒`
}

/** 从 OSS 直链 / 旧应用 URL / 裸对象键里提取对象键。 */
function extractKey(raw: string, overview: OssOverview | null): string | null {
  const text = raw.trim()
  if (!text) return null
  if (text.startsWith('/') || /^https?:\/\//i.test(text)) {
    try {
      const parsed = new URL(text, window.location.origin)
      const marker = '/api/oss/images/'
      const index = parsed.pathname.indexOf(marker)
      if (index >= 0) {
        return decodeURIComponent(parsed.pathname.slice(index + marker.length)).replace(/^\/+/, '')
      }
      const endpointHost = overview?.endpoint ? new URL(overview.endpoint).host : ''
      if (endpointHost && parsed.host === endpointHost) {
        return decodeURIComponent(parsed.pathname).replace(/^\/+/, '')
      }
      if (!endpointHost && parsed.host.endsWith('.aliyuncs.com')) {
        return decodeURIComponent(parsed.pathname).replace(/^\/+/, '')
      }
      return null
    } catch {
      return null
    }
  }
  return text.replace(/^\/+/, '')
}

/** 私域 OSS：上传 / 列举 / 删除走本服务，图片统一用 OSS 直链。 */
export function OssPage() {
  const { message } = App.useApp()
  const [overview, setOverview] = useState<OssOverview | null>(null)
  const [overviewError, setOverviewError] = useState('')
  const [objects, setObjects] = useState<OssObject[]>([])
  const [listLoading, setListLoading] = useState(false)
  const [listError, setListError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploaded, setUploaded] = useState<OssImage | null>(null)
  const [urlInput, setUrlInput] = useState('')
  const [resolving, setResolving] = useState(false)
  const [preview, setPreview] = useState<PreviewImage | null>(null)
  const [deleting, setDeleting] = useState('')

  const loadOverview = useCallback(async () => {
    try {
      setOverview(await api.ossOverview())
      setOverviewError('')
    } catch (error) {
      setOverviewError(errorText(error))
    }
  }, [])

  const loadObjects = useCallback(async () => {
    setListLoading(true)
    setListError('')
    try {
      const next = await api.ossObjects()
      setObjects(next.objects)
    } catch (error) {
      setListError(errorText(error))
      setObjects([])
    } finally {
      setListLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadOverview()
  }, [loadOverview])

  useEffect(() => {
    // overview 请求会由后端签发 HttpOnly 会话 Cookie；拿到配置后再管理对象。
    if (overview?.access_configured) void loadObjects()
  }, [overview?.access_configured, loadObjects])

  const copy = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text)
      message.success(`${label}已复制`)
    } catch {
      message.error('复制失败，请手动选择文本复制')
    }
  }

  const showImage = (image: PreviewImage) => {
    setPreview(image)
    setUrlInput(image.url)
  }

  const handleUpload = async (file: File) => {
    setUploading(true)
    try {
      const result = await api.ossUpload(file)
      setUploaded(result)
      showImage({
        key: result.key,
        url: result.url,
        downloadUrl: result.download_url,
        expiresIn: result.expires_in,
      })
      message.success('上传成功，已生成 OSS 直链')
      void loadObjects()
    } catch (error) {
      message.error(errorText(error))
    } finally {
      setUploading(false)
    }
  }

  const beforeUpload: UploadProps['beforeUpload'] = (file) => {
    const suffix = `.${(file.name.split('.').pop() ?? '').toLowerCase()}`
    if (overview && !overview.accepted_suffixes.includes(suffix)) {
      message.error(`只支持 ${overview.accepted_suffixes.join(' / ')}`)
      return Upload.LIST_IGNORE
    }
    if (overview && file.size > overview.max_file_mb * 1024 * 1024) {
      message.error(`图片不能超过 ${overview.max_file_mb} MB`)
      return Upload.LIST_IGNORE
    }
    void handleUpload(file)
    return Upload.LIST_IGNORE
  }

  const openUrl = async (raw: string) => {
    const key = extractKey(raw, overview)
    if (!key) {
      message.warning('请粘贴本 Bucket 的 OSS 直链，或 image3/2026/11/05/xxx.png 对象键')
      return
    }
    setResolving(true)
    try {
      // 统一走 OSS 直链：按对象键重新换取预览 / 下载两个直链。
      const result = await api.ossPresign(key)
      showImage({
        key: result.key,
        url: result.url,
        downloadUrl: result.download_url,
        expiresIn: result.expires_in,
      })
      message.success('已换取新的 OSS 直链')
    } catch (error) {
      message.error(errorText(error))
    } finally {
      setResolving(false)
    }
  }

  const handleDelete = async (key: string) => {
    setDeleting(key)
    try {
      await api.ossDelete(key)
      message.success('已删除')
      if (uploaded?.key === key) {
        setUploaded(null)
        setUrlInput('')
      }
      if (preview?.key === key) setPreview(null)
      void loadObjects()
    } catch (error) {
      message.error(errorText(error))
    } finally {
      setDeleting('')
    }
  }

  const columns: TableProps<OssObject>['columns'] = [
    {
      title: '预览',
      dataIndex: 'url',
      width: 92,
      render: (url: string) => (
        <Image
          src={url}
          width={52}
          height={52}
          style={{ objectFit: 'cover', borderRadius: 8 }}
          preview={{ src: url }}
          fallback="data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='52' height='52'%3E%3Crect width='52' height='52' fill='%23eef2f7'/%3E%3Ctext x='50%25' y='54%25' font-size='10' text-anchor='middle' fill='%23888'%3Eno%20img%3C/text%3E%3C/svg%3E"
        />
      ),
    },
    {
      title: '对象键',
      dataIndex: 'key',
      render: (key: string, record) => (
        <Space orientation="vertical" size={0} style={{ maxWidth: 420 }}>
          <Typography.Text strong ellipsis={{ tooltip: record.name }}>
            {record.name}
          </Typography.Text>
          <Typography.Text
            type="secondary"
            className="mono-text"
            ellipsis={{ tooltip: key }}
            copyable={{ text: key, tooltips: ['复制对象键', '已复制'] }}
            style={{ maxWidth: 420 }}
          >
            {key}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: '大小',
      dataIndex: 'size',
      width: 100,
      align: 'right',
      render: (size: number) => formatBytes(size),
    },
    {
      title: '修改时间',
      dataIndex: 'last_modified',
      width: 164,
      render: (value: string | null) =>
        value ? dayjs(value).format('YYYY-MM-DD HH:mm') : '—',
    },
    {
      title: '操作',
      key: 'actions',
      width: 232,
      render: (_, record) => (
        <Space size={0}>
          <Tooltip title="预览这张图片">
            <Button
              type="link"
              size="small"
              icon={<LinkOutlined />}
              onClick={() =>
                showImage({
                  key: record.key,
                  url: record.url,
                  downloadUrl: record.download_url,
                  expiresIn: record.expires_in,
                })
              }
            >
              访问
            </Button>
          </Tooltip>
          <Tooltip title="复制 OSS 直链">
            <Button
              type="link"
              size="small"
              icon={<CopyOutlined />}
              onClick={() => void copy(record.url, 'OSS 直链')}
            />
          </Tooltip>
          <Tooltip title="下载">
            <Button
              type="link"
              size="small"
              icon={<DownloadOutlined />}
              href={record.download_url}
              target="_blank"
              rel="noreferrer"
            />
          </Tooltip>
          <Popconfirm
            title="删除这张图片？"
            description="删除后直链立即失效，且不可恢复。"
            okText="删除"
            okButtonProps={{ danger: true }}
            cancelText="取消"
            onConfirm={() => void handleDelete(record.key)}
          >
            <Button
              type="link"
              size="small"
              danger
              icon={<DeleteOutlined />}
              loading={deleting === record.key}
            />
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
            私域 OSS
          </Typography.Title>
          <Typography.Text type="secondary">
            阿里云私有 Bucket：上传即返回 OSS 直链，粘贴直链即可预览与下载
          </Typography.Text>
        </div>
        <Space>
          <Tag color="purple">私有 Bucket · OSS 直链</Tag>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => {
              void loadOverview()
              if (overview?.access_configured) void loadObjects()
            }}
          >
            刷新
          </Button>
        </Space>
      </div>

      {overviewError && (
        <Alert type="error" showIcon message="无法读取 OSS 配置" description={overviewError} />
      )}
      {overview?.warnings.map((warning) => (
        <Alert key={warning} type="warning" showIcon message={warning} />
      ))}

      <Row gutter={[16, 16]} className="stat-row">
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic title="Bucket" value={overview?.bucket ?? '—'} />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic title="目录前缀" value={overview?.prefix ?? '—'} />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic title="图片读取" value="OSS 直链" />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card className="stat-card">
            <Statistic
              title="直链有效期"
              value={overview ? formatExpiry(overview.url_expires) : '—'}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}>
          <Card
            className="panel"
            title={
              <Space>
                <CloudUploadOutlined />
                上传图片
              </Space>
            }
            extra={
              <Tag>
                {overview
                  ? `${overview.accepted_suffixes.join(' ')} · ≤ ${overview.max_file_mb} MB`
                  : '读取配置中'}
              </Tag>
            }
          >
            <div className="upload-dragger-wrap">
              <Upload.Dragger
                accept={overview?.accepted_suffixes.join(',')}
                maxCount={1}
                showUploadList={false}
                beforeUpload={beforeUpload}
                disabled={!overview?.configured || !overview.access_configured || uploading}
              >
                <p className="ant-upload-drag-icon">
                  <CloudUploadOutlined />
                </p>
                <p className="ant-upload-text">
                  {uploading ? '正在上传到私有 Bucket…' : '点击或拖拽图片到这里上传'}
                </p>
                <p className="ant-upload-hint">
                  上传成功后直接返回 OSS 直链，可预览、复制、访问与下载
                </p>
              </Upload.Dragger>
            </div>

            {uploaded && (
              <div className="oss-upload-result">
                <Image
                  src={uploaded.url}
                  width={132}
                  height={132}
                  className="oss-result-image"
                  style={{ objectFit: 'cover', borderRadius: 10 }}
                />
                <div className="oss-result-meta">
                  <Descriptions
                    size="small"
                    column={1}
                    items={[
                      { key: 'name', label: '文件', children: uploaded.name },
                      {
                        key: 'key',
                        label: '对象键',
                        children: (
                          <Typography.Text
                            className="mono-text"
                            copyable={{ text: uploaded.key, tooltips: ['复制对象键', '已复制'] }}
                          >
                            {uploaded.key}
                          </Typography.Text>
                        ),
                      },
                      { key: 'size', label: '大小', children: formatBytes(uploaded.size) },
                      {
                        key: 'expires',
                        label: '直链有效期',
                        children: formatExpiry(uploaded.expires_in),
                      },
                    ]}
                  />
                  <Space.Compact style={{ width: '100%', marginTop: 8 }}>
                    <Input readOnly value={uploaded.url} />
                    <Button
                      icon={<CopyOutlined />}
                      onClick={() => void copy(uploaded.url, 'OSS 直链')}
                    >
                      复制
                    </Button>
                  </Space.Compact>
                  <Space wrap style={{ marginTop: 10 }}>
                    <Button
                      type="primary"
                      ghost
                      icon={<LinkOutlined />}
                      onClick={() =>
                        showImage({
                          key: uploaded.key,
                          url: uploaded.url,
                          downloadUrl: uploaded.download_url,
                          expiresIn: uploaded.expires_in,
                        })
                      }
                    >
                      预览
                    </Button>
                    <Button
                      icon={<DownloadOutlined />}
                      href={uploaded.download_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      下载
                    </Button>
                    <Popconfirm
                      title="删除这张图片？"
                      description="删除后直链立即失效，且不可恢复。"
                      okText="删除"
                      okButtonProps={{ danger: true }}
                      cancelText="取消"
                      onConfirm={() => void handleDelete(uploaded.key)}
                    >
                      <Button danger icon={<DeleteOutlined />} loading={deleting === uploaded.key}>
                        删除
                      </Button>
                    </Popconfirm>
                  </Space>
                </div>
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card
            className="panel"
            title={
              <Space>
                <SearchOutlined />
                访问 OSS 直链
              </Space>
            }
            extra={<Tag>OSS 直链 / 对象键</Tag>}
          >
            <Space.Compact style={{ width: '100%' }}>
              <Input
                value={urlInput}
                onChange={(event) => setUrlInput(event.target.value)}
                onPressEnter={() => void openUrl(urlInput)}
                allowClear
                placeholder="粘贴 OSS 直链，或 image3/2026/11/05/xxx.png"
              />
              <Button
                type="primary"
                icon={<SearchOutlined />}
                loading={resolving}
                onClick={() => void openUrl(urlInput)}
              >
                访问
              </Button>
            </Space.Compact>
            <Typography.Paragraph type="secondary" style={{ margin: '10px 0 0', fontSize: 12 }}>
              粘贴 OSS 直链或裸对象键，服务端会按 key 重新换取预览 / 下载直链；链接过期后
              点「访问」即可刷新。
            </Typography.Paragraph>

            {preview ? (
              <div className="oss-preview">
                <Image src={preview.url} className="oss-preview-image" />
                <div className="oss-preview-meta">
                  <Typography.Text className="mono-text" ellipsis={{ tooltip: preview.key }}>
                    {preview.key}
                  </Typography.Text>
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    直链有效期 {formatExpiry(preview.expiresIn)}
                  </Typography.Text>
                  <Space wrap>
                    <Button
                      icon={<CopyOutlined />}
                      onClick={() => void copy(preview.url, 'OSS 直链')}
                    >
                      复制直链
                    </Button>
                    <Button
                      type="primary"
                      icon={<DownloadOutlined />}
                      href={preview.downloadUrl}
                      target="_blank"
                      rel="noreferrer"
                    >
                      下载图片
                    </Button>
                    <Popconfirm
                      title="删除这张图片？"
                      description="删除后直链立即失效，且不可恢复。"
                      okText="删除"
                      okButtonProps={{ danger: true }}
                      cancelText="取消"
                      onConfirm={() => void handleDelete(preview.key)}
                    >
                      <Button danger icon={<DeleteOutlined />} loading={deleting === preview.key}>
                        删除
                      </Button>
                    </Popconfirm>
                  </Space>
                </div>
              </div>
            ) : (
              <div className="oss-preview oss-preview-empty">
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="粘贴 OSS 直链或对象键后点「访问」，图片会预览在这里"
                />
              </div>
            )}
          </Card>
        </Col>
      </Row>

      <Card
        className="panel"
        title={
          <Space>
            <CloudServerOutlined />
            最近上传
          </Space>
        }
        extra={
          <Space>
            <Tag>{objects.length} 个对象</Tag>
            <Button
              icon={<ReloadOutlined />}
              loading={listLoading}
              onClick={() => void loadObjects()}
            >
              刷新
            </Button>
          </Space>
        }
      >
        {listError && (
          <Alert
            type={listError.includes('access') || listError.includes('口令') ? 'warning' : 'error'}
            showIcon
            message="无法读取图片列表"
            description={listError}
            style={{ marginBottom: 12 }}
          />
        )}
        <Table<OssObject>
          rowKey="key"
          size="middle"
          loading={listLoading}
          columns={columns}
          dataSource={objects}
          scroll={{ x: 980 }}
          pagination={{ pageSize: 8, hideOnSinglePage: true, showSizeChanger: false }}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={
                  overview?.access_configured
                    ? '还没有图片，上传一张试试'
                    : '服务端未配置 OSS_ACCESS_TOKEN，暂不能读写私有 Bucket'
                }
              />
            ),
          }}
        />
        <div className="meta-line">
          私有 Bucket <code>{overview?.bucket ?? '—'}</code> · 目录{' '}
          <code>{overview?.prefix ?? '—'}/</code> · 图片统一走 OSS 直链，链接带{' '}
          <code>OSSAccessKeyId</code> + 限时签名，过期后重新访问即可
        </div>
      </Card>
    </div>
  )
}
