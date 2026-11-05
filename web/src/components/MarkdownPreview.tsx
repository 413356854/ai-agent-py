import type { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { resolveAsset } from '../api/client'

interface Props {
  markdown: string
  assetsBase: string
  children?: ReactNode
  /** 点击答案里的引用编号（`[n]`）时回调；不传则按普通链接渲染。 */
  onCitationClick?: (index: number) => void
}

/** 渲染 MinerU 输出的 Markdown（GFM 表格 + 相对路径图片 + 可点击引用）。 */
export function MarkdownPreview({ markdown, assetsBase, onCitationClick }: Props) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          img: ({ node: _node, src, alt, ...rest }) => (
            <img
              src={src ? resolveAsset(assetsBase, src) : undefined}
              alt={alt ?? ''}
              loading="lazy"
              {...rest}
            />
          ),
          a: ({ node: _node, href, children, ...rest }) => {
            const citation = href ? /^#rag-cite-(\d+)$/.exec(href) : null
            if (citation && onCitationClick) {
              const index = Number(citation[1])
              return (
                <sup
                  className="cite-ref"
                  role="button"
                  tabIndex={0}
                  title={`查看来源 [${index}]`}
                  onClick={(event) => {
                    event.preventDefault()
                    onCitationClick?.(index)
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      onCitationClick?.(index)
                    }
                  }}
                >
                  {children}
                </sup>
              )
            }
            return (
              <a href={href} target="_blank" rel="noreferrer" {...rest}>
                {children}
              </a>
            )
          },
          table: ({ node: _node, children, ...rest }) => (
            <div className="table-scroll">
              <table {...rest}>{children}</table>
            </div>
          ),
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  )
}
