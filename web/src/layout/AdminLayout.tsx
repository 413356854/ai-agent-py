import {
  AppstoreOutlined,
  AuditOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  FileSearchOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  MessageOutlined,
  RobotOutlined,
  TableOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { Breadcrumb, Button, Layout, Menu, Space, Tag, Tooltip, Typography } from 'antd'
import type { MenuProps } from 'antd'
import { useEffect, useState } from 'react'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { ThemeToggle } from '../components/ThemeToggle'

const { Header, Sider, Content } = Layout

const SIDER_COLLAPSED_KEY = 'ai-agent-py:sider-collapsed'

const MENU_ITEMS: MenuProps['items'] = [
  {
    key: 'rag',
    icon: <DatabaseOutlined />,
    label: 'RAG 管理',
    children: [
      { key: '/rag/documents', icon: <FileSearchOutlined />, label: '文档解析' },
      { key: '/rag/records', icon: <TableOutlined />, label: 'RAG 记录' },
      { key: '/rag/chat', icon: <MessageOutlined />, label: 'RAG 问答' },
    ],
  },
  {
    key: 'agent',
    icon: <AppstoreOutlined />,
    label: 'Agent 应用',
    children: [
      { key: '/agents', icon: <RobotOutlined />, label: '应用列表' },
      { key: '/agents/reimburse', icon: <AuditOutlined />, label: '报销审核' },
      { key: '/agents/general', icon: <ThunderboltOutlined />, label: '通用 Agent' },
    ],
  },
  {
    key: '/oss',
    icon: <CloudServerOutlined />,
    label: '私域 OSS',
  },
]

/** 路由 → 面包屑标题；未登记的路径回落到文档解析。 */
const PAGE_TITLES: Record<string, string> = {
  '/rag/documents': '文档解析',
  '/rag/records': 'RAG 记录',
  '/rag/chat': 'RAG 问答',
  '/agents': '应用列表',
  '/agents/reimburse': '报销审核',
  '/agents/general': '通用 Agent',
  '/oss': '图片上传与访问',
}

function readSiderCollapsed(): boolean {
  try {
    return window.localStorage.getItem(SIDER_COLLAPSED_KEY) === '1'
  } catch {
    return false
  }
}

/** 管理后台框架：左侧栏目树 + 顶部面包屑 + 内容区。 */
export function AdminLayout() {
  const location = useLocation()
  const navigate = useNavigate()
  const selectedKeys = [location.pathname]
  const isAgentSection = location.pathname.startsWith('/agents')
  const isOssSection = location.pathname.startsWith('/oss')
  const sectionTitle = isOssSection ? '私域 OSS' : isAgentSection ? 'Agent 应用' : 'RAG 管理'
  const pageTitle =
    PAGE_TITLES[location.pathname] ??
    (isOssSection ? '图片上传与访问' : isAgentSection ? '应用列表' : '文档解析')
  const [collapsed, setCollapsed] = useState(readSiderCollapsed)
  const toggleLabel = collapsed ? '展开侧栏' : '收起侧栏'

  useEffect(() => {
    try {
      window.localStorage.setItem(SIDER_COLLAPSED_KEY, collapsed ? '1' : '0')
    } catch {
      // localStorage 不可用（如隐私模式）时只保留当前会话状态。
    }
  }, [collapsed])

  return (
    <Layout className="admin-layout">
      <Sider
        width={232}
        collapsedWidth={0}
        collapsible
        collapsed={collapsed}
        trigger={null}
        theme="light"
        className="admin-sider"
        inert={collapsed || undefined}
      >
        <div className="brand">
          <div className="brand-logo">AI</div>
          {!collapsed && (
            <div className="brand-text">
              <div className="brand-title">ai-agent-py</div>
              <div className="brand-sub">RAG · Agent 控制台</div>
            </div>
          )}
        </div>
        <Menu
          mode="inline"
          items={MENU_ITEMS}
          selectedKeys={selectedKeys}
          defaultOpenKeys={['rag', 'agent']}
          onClick={({ key }) => {
            if (key.startsWith('/')) navigate(key)
          }}
          style={{ borderInlineEnd: 'none' }}
        />
        {!collapsed && (
          <div className="sider-footer">
            <div>ai-agent-py v0.1.0</div>
            <div>MinerU · Milvus · OSS</div>
          </div>
        )}
      </Sider>

      <Layout>
        <Header className="admin-header">
          <div className="header-left">
            <Tooltip title={toggleLabel} placement="bottomLeft">
              <Button
                type="text"
                className="sider-toggle"
                aria-label={toggleLabel}
                aria-expanded={!collapsed}
                icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                onClick={() => setCollapsed((value) => !value)}
              />
            </Tooltip>
            <Breadcrumb items={[{ title: sectionTitle }, { title: pageTitle }]} />
          </div>
          <Space size={12}>
            <Tag color="geekblue" className="header-tag">
              Milvus · LangGraph
            </Tag>
            <Typography.Link href="/docs" target="_blank" rel="noreferrer">
              API 文档
            </Typography.Link>
            <ThemeToggle />
          </Space>
        </Header>
        <Content className="admin-content">
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
