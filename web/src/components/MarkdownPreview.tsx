import type { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { resolveAsset } from '../api/client'

interface Props {
  markdown: string
  assetsBase: string
  children?: ReactNode
}

/** 渲染 MinerU 输出的 Markdown（GFM 表格 + 相对路径图片）。 */
export function MarkdownPreview({ markdown, assetsBase }: Props) {
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
          a: ({ node: _node, href, children, ...rest }) => (
            <a href={href} target="_blank" rel="noreferrer" {...rest}>
              {children}
            </a>
          ),
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
