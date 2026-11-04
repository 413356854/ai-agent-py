import { useEffect, useState } from 'react'
import {
  Alert,
  App,
  Button,
  Descriptions,
  Empty,
  InputNumber,
  Modal,
  Space,
  Spin,
  Tag,
  Typography,
} from 'antd'
import {
  DatabaseOutlined,
  DownOutlined,
  ReloadOutlined,
  SplitCellsOutlined,
} from '@ant-design/icons'
import { ApiError, api } from '../api/client'
import type { ChunkOut, ParseJob, RagIngestResult, SplitPreview } from '../types'

interface SplitPreviewModalProps {
  open: boolean
  job: ParseJob | null
  onClose: () => void
  onIngested: (job: ParseJob) => void
}

const DEFAULT_CHUNK_SIZE = 800
const DEFAULT_CHUNK_OVERLAP = 120
// 与后端一致：预览最多拿 5000 个片段，足够覆盖常见文档；缓存里始终是完整切片。
const PREVIEW_LIMIT = 5000
// 一次渲染多少个片段：几千个 <pre> 同时挂到 DOM 上会明显卡顿。
const PAGE_SIZE = 100

const MODE_LABELS: Record<RagIngestResult['mode'], string> = {
  cached: '按切分缓存',
  reviewed: '按回传切片',
  resplit: '服务端重新切分',
}

function errorText(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return String(error)
}

/** JSON 片段在界面上展开成缩进格式，方便阅读；入库的仍是紧凑 JSON。 */
function displayContent(chunk: ChunkOut, strategy: string): string {
  if (strategy !== 'json') return chunk.content
  try {
    return JSON.stringify(JSON.parse(chunk.content), null, 2)
  } catch {
    return chunk.content
  }
}

function headerTags(chunk: ChunkOut) {
  return (['h1', 'h2', 'h3', 'h4'] as const)
    .filter((key) => chunk.metadata[key] !== undefined)
    .map((key) => (
      <Tag key={key} color="geekblue" style={{ marginInlineEnd: 0 }}>
        {String(chunk.metadata[key])}
      </Tag>
    ))
}

export function SplitPreviewModal({ open, job, onClose, onIngested }: SplitPreviewModalProps) {
  const { message } = App.useApp()
  const [preview, setPreview] = useState<SplitPreview | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [chunkSize, setChunkSize] = useState(DEFAULT_CHUNK_SIZE)
  const [chunkOverlap, setChunkOverlap] = useState(DEFAULT_CHUNK_OVERLAP)
  const [ingesting, setIngesting] = useState(false)
  const [ingested, setIngested] = useState<RagIngestResult | null>(null)
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE)

  const load = async (target: ParseJob, size?: number, overlap?: number) => {
    setLoading(true)
    setError('')
    setIngested(null)
    try {
      const data = await api.split(target.id, {
        // 首次打开不传块参数，用后端 .env 里的 CHUNK_SIZE / CHUNK_OVERLAP。
        ...(size !== undefined ? { chunk_size: size } : {}),
        ...(overlap !== undefined ? { chunk_overlap: overlap } : {}),
        limit: PREVIEW_LIMIT,
      })
      setPreview(data)
      setChunkSize(data.chunk_size)
      setChunkOverlap(data.chunk_overlap)
      setVisibleCount(PAGE_SIZE)
    } catch (loadError) {
      setPreview(null)
      setError(errorText(loadError))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!open || !job) return
    setPreview(null)
    setError('')
    setIngested(null)
    void load(job)
    // 只在弹窗打开 / 换文档时重新切分；改参数由「重新切分」按钮触发。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, job?.id])

  /**
   * 入库：只把切分预览返回的 split_id 回传，后端按 id 从 HashMap 缓存读取
   * 完整切片——既不重新切分，也不用把几千条切片塞进请求体。
   */
  const handleIngest = async () => {
    if (!job || !preview) return
    if (!preview.chunks.length) {
      message.warning('没有可入库的切片')
      return
    }
    setIngesting(true)
    try {
      const result = await api.ingestRag(job.id, {
        split_id: preview.split_id,
        // 让后端校验「缓存里的切片数 = 预览总数」，防止切分结果前后不一致
        total: preview.total,
      })
      setIngested(result)
      message.success(
        `已写入 ${result.chunks} 个切片到 ${result.database}.${result.collection}（${MODE_LABELS[result.mode]}）`,
      )
      onIngested(job)
    } catch (ingestError) {
      if (ingestError instanceof ApiError && ingestError.status === 409) {
        // 缓存过期 / 服务重启：自动重新切分，换一个可入库的 split_id
        message.error(`${errorText(ingestError)} 已自动重新切分，请确认后再点「入库 RAG」`)
        void load(job, chunkSize, chunkOverlap)
      } else {
        message.error(errorText(ingestError))
      }
    } finally {
      setIngesting(false)
    }
  }

  const overflow = Boolean(preview?.truncated)
  const disabled = loading || ingesting || !preview || !preview.chunks.length
  const visibleChunks = preview ? preview.chunks.slice(0, visibleCount) : []

  return (
    <Modal
      open={open && Boolean(job)}
      onCancel={onClose}
      width={1000}
      destroyOnHidden
      maskClosable={!ingesting}
      title={
        <Space>
          <SplitCellsOutlined />
          切分预览 · {job?.file_name ?? ''}
        </Space>
      }
      footer={
        <div className="split-footer">
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {preview
              ? overflow
                ? `共 ${preview.total} 个片段 · 仅展示 ${preview.chunks.length} 个 · 入库将写入缓存的全部 ${preview.total} 个（重复入库先删除旧切片）`
                : `共 ${preview.total} 个片段 · ${preview.characters.toLocaleString()} 字符 · 将把缓存的 ${preview.total} 个切片写入 Milvus（重复入库先删除旧切片）`
              : '读取解析结果 Markdown 切分'}
          </Typography.Text>
          <Space>
            <Button onClick={onClose} disabled={ingesting}>
              关闭
            </Button>
            <Button
              type="primary"
              icon={<DatabaseOutlined />}
              loading={ingesting}
              disabled={disabled}
              onClick={() => void handleIngest()}
            >
              {ingested || job?.rag_ingested ? '重新入库 RAG' : '入库 RAG'}
            </Button>
          </Space>
        </div>
      }
    >
      <div className="split-toolbar">
        <Space size={6} wrap>
          <Tag color="blue">{preview?.strategy_label ?? '自动选择切分器'}</Tag>
          {preview?.source_file && <Tag color="cyan">来源 {preview.source_file}</Tag>}
          {preview && <Tag color="green">缓存 {preview.split_id.slice(0, 8)}</Tag>}
          <Tag>{job?.suffix ? `.${job.suffix}` : '未知后缀'}</Tag>
          {preview && <Tag color="purple">块 {preview.chunk_size}</Tag>}
          {preview && <Tag color="purple">重叠 {preview.chunk_overlap}</Tag>}
        </Space>
        <Space size={6} wrap>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            块大小
          </Typography.Text>
          <InputNumber
            min={20}
            max={4000}
            step={100}
            value={chunkSize}
            disabled={loading || ingesting}
            style={{ width: 92 }}
            onChange={(value) => setChunkSize(value ?? DEFAULT_CHUNK_SIZE)}
          />
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            重叠
          </Typography.Text>
          <InputNumber
            min={0}
            max={1000}
            step={20}
            value={chunkOverlap}
            disabled={loading || ingesting}
            style={{ width: 82 }}
            onChange={(value) => setChunkOverlap(value ?? DEFAULT_CHUNK_OVERLAP)}
          />
          <Button
            icon={<ReloadOutlined />}
            loading={loading}
            disabled={ingesting || !job}
            onClick={() => job && void load(job, chunkSize, chunkOverlap)}
          >
            重新切分
          </Button>
        </Space>
      </div>

      {error && (
        <Alert
          type="error"
          showIcon
          message="切分失败"
          description={error}
          style={{ marginTop: 12 }}
        />
      )}

      {loading && (
        <div className="split-loading">
          <Spin size="large" />
          <span>正在切分…</span>
        </div>
      )}

      {!loading && !error && preview && (
        <>
          <Descriptions
            size="small"
            column={4}
            style={{ marginTop: 12 }}
            items={[
              { key: 'total', label: '片段数', children: preview.total },
              {
                key: 'chars',
                label: '总字符',
                children: preview.characters.toLocaleString(),
              },
              {
                key: 'ingest',
                label: '待入库',
                children: preview.total,
              },
              {
                key: 'source',
                label: '切分来源',
                children: preview.source_file || '未生成',
              },
            ]}
          />
          {overflow && (
            <Alert
              type="info"
              showIcon
              style={{ marginTop: 8 }}
              message={`片段较多，仅展示前 ${preview.chunks.length} 个（共 ${preview.total} 个）；入库会按缓存写入全部 ${preview.total} 个切片。`}
            />
          )}
          {job?.rag_ingested && !ingested && (
            <Alert
              type="info"
              showIcon
              style={{ marginTop: 8 }}
              message={`该文档已入库 RAG（${job.rag_ingest?.chunks ?? 0} 个切片，${job.rag_ingest?.collection ?? 'Milvus'}）。重复入库会先删除该文档的旧切片，再写入本次切片。`}
            />
          )}
          {ingested && (
            <Alert
              type="success"
              showIcon
              style={{ marginTop: 8 }}
              message={`已入库：${ingested.chunks} 个切片 → ${ingested.database}.${ingested.collection}（${ingested.elapsed_seconds}s，${MODE_LABELS[ingested.mode]}）`}
            />
          )}

          {preview.chunks.length === 0 ? (
            <Empty
              style={{ margin: '32px 0' }}
              description="没有切分出内容，请检查文档正文"
            />
          ) : (
            <div className="split-list">
              {visibleChunks.map((chunk) => (
                <div className="split-chunk" key={chunk.index}>
                  <div className="split-chunk-head">
                    <Space size={6} wrap>
                      <span className="split-chunk-index">#{chunk.index + 1}</span>
                      {headerTags(chunk)}
                      {chunk.metadata.page !== undefined && (
                        <Tag style={{ marginInlineEnd: 0 }}>第 {chunk.metadata.page} 页</Tag>
                      )}
                    </Space>
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      {chunk.characters} 字符
                    </Typography.Text>
                  </div>
                  <pre className="split-chunk-content">
                    {displayContent(chunk, preview.strategy)}
                  </pre>
                </div>
              ))}
              {visibleChunks.length < preview.chunks.length && (
                <Button
                  block
                  icon={<DownOutlined />}
                  onClick={() => setVisibleCount((count) => count + PAGE_SIZE)}
                >
                  显示更多（剩余 {preview.chunks.length - visibleChunks.length} 个，入库仍包含全部{' '}
                  {preview.chunks.length} 个）
                </Button>
              )}
            </div>
          )}
        </>
      )}

      {!loading && !error && !preview && (
        <Empty style={{ margin: '32px 0' }} description="暂无切分结果" />
      )}
    </Modal>
  )
}
