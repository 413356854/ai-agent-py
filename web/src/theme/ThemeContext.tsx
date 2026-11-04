import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

export type ThemeMode = 'light' | 'dark'

const STORAGE_KEY = 'ai-agent-py:theme'

interface ThemeContextValue {
  mode: ThemeMode
  setMode: (mode: ThemeMode) => void
  toggle: () => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)

/** 初始主题：localStorage > index.html 里预置的 data-theme > 系统偏好。 */
function initialMode(): ThemeMode {
  if (typeof window === 'undefined') return 'light'
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    // 隐私模式下 localStorage 可能不可用
  }
  const preset = document.documentElement.dataset.theme
  if (preset === 'light' || preset === 'dark') return preset
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

/** 亮/暗主题状态：写 `<html data-theme>`、`color-scheme` 并持久化到 localStorage。 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>(initialMode)

  useEffect(() => {
    const root = document.documentElement
    root.dataset.theme = mode
    root.style.colorScheme = mode
    try {
      window.localStorage.setItem(STORAGE_KEY, mode)
    } catch {
      // 存不了就只影响下次打开时的默认值
    }
  }, [mode])

  const setMode = useCallback((next: ThemeMode) => setModeState(next), [])
  const toggle = useCallback(
    () => setModeState((previous) => (previous === 'dark' ? 'light' : 'dark')),
    [],
  )
  const value = useMemo(() => ({ mode, setMode, toggle }), [mode, setMode, toggle])

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeContextValue {
  const value = useContext(ThemeContext)
  if (!value) throw new Error('useTheme 必须在 ThemeProvider 内使用')
  return value
}
