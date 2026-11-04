import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App as AntdApp, ConfigProvider, theme as antdTheme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import dayjs from 'dayjs'
import 'dayjs/locale/zh-cn'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ThemeProvider, useTheme } from './theme/ThemeContext'
import './styles.css'

dayjs.locale('zh-cn')

const FONT_FAMILY =
  '-apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif'

/** 主题要跟随 ThemeProvider，因此 ConfigProvider 放在组件里而不是模块顶层。 */
function Root() {
  const { mode } = useTheme()
  const isDark = mode === 'dark'

  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: isDark ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        token: {
          colorPrimary: '#1677ff',
          borderRadius: 8,
          fontFamily: FONT_FAMILY,
        },
        components: isDark
          ? {
              Layout: {
                headerBg: '#141b26',
                siderBg: '#141b26',
                lightSiderBg: '#141b26',
                bodyBg: '#0b0f17',
              },
              Menu: {
                itemBorderRadius: 8,
                itemMarginInline: 10,
                itemBg: 'transparent',
                subMenuItemBg: 'transparent',
              },
              Card: { colorBgContainer: '#141b26' },
              Table: { headerBg: '#18212e', rowHoverBg: '#1a2432' },
            }
          : {
              Layout: { headerBg: '#ffffff', siderBg: '#ffffff', bodyBg: '#f4f6fa' },
              Menu: { itemBorderRadius: 8, itemMarginInline: 10 },
            },
      }}
    >
      <AntdApp>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </AntdApp>
    </ConfigProvider>
  )
}

createRoot(document.getElementById('root') as HTMLElement).render(
  <StrictMode>
    <ThemeProvider>
      <Root />
    </ThemeProvider>
  </StrictMode>,
)
