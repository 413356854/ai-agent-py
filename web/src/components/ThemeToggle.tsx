import { MoonOutlined, SunOutlined } from '@ant-design/icons'
import { Button, Tooltip } from 'antd'
import { useTheme } from '../theme/ThemeContext'

/** 右上角亮/暗主题切换按钮。 */
export function ThemeToggle() {
  const { mode, toggle } = useTheme()
  const isDark = mode === 'dark'
  const label = isDark ? '切换到亮色模式' : '切换到暗色模式'

  return (
    <Tooltip title={label}>
      <Button
        type="text"
        shape="circle"
        aria-label={label}
        title={label}
        icon={isDark ? <SunOutlined /> : <MoonOutlined />}
        onClick={toggle}
      />
    </Tooltip>
  )
}
