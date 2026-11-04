import { useEffect, useRef, useState } from 'react'
import { Alert, Spin } from 'antd'
import mermaid from 'mermaid'
import { useTheme } from '../theme/ThemeContext'

let initialized = false
let sequence = 0

const FONT_FAMILY =
  '-apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif'

function ensureInit() {
  if (initialized) return
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'loose',
    theme: 'base',
    fontFamily: FONT_FAMILY,
    flowchart: { curve: 'basis', htmlLabels: true, useMaxWidth: true },
  })
  initialized = true
}

/** 深色主题下替换 LangGraph 导出的 classDef 配色（只改颜色，不动拓扑与标签）。 */
function tune(code: string, dark: boolean): string {
  if (!dark) return code
  return code
    .replace(
      /classDef default fill:#f2f0ff[^\n]*/,
      'classDef default fill:#1b2230,color:#e6edf3,stroke:#3b4657,line-height:1.2',
    )
    .replace(/classDef last fill:#bfb6fc[^\n]*/, 'classDef last fill:#3a3358,color:#e6edf3')
    .replace(/classDef first fill-opacity:0[^\n]*/, 'classDef first fill-opacity:0,color:#9aa4b2')
}

interface MermaidDiagramProps {
  /** LangGraph `draw_mermaid()` 的原始输出。 */
  code: string
  minHeight?: number
}

/** 渲染 mermaid 流程图；主题切换时重绘。 */
export function MermaidDiagram({ code, minHeight = 320 }: MermaidDiagramProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const { mode } = useTheme()
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    const draw = async () => {
      ensureInit()
      setLoading(true)
      sequence += 1
      try {
        const { svg } = await mermaid.render(
          `agent-graph-${sequence}`,
          tune(code, mode === 'dark'),
        )
        if (cancelled || !hostRef.current) return
        hostRef.current.innerHTML = svg
        setError('')
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void draw()
    return () => {
      cancelled = true
    }
  }, [code, mode])

  if (error) {
    return <Alert type="error" showIcon message="流程图渲染失败" description={error} />
  }

  return (
    <div className="agent-graph" style={{ minHeight }}>
      <div ref={hostRef} className="agent-graph-canvas" />
      {loading && (
        <div className="agent-graph-loading">
          <Spin />
        </div>
      )}
    </div>
  )
}
