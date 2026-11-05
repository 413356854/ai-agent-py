import { Navigate, Route, Routes } from 'react-router-dom'
import { AdminLayout } from './layout/AdminLayout'
import { AgentDetailPage } from './pages/agent/AgentDetailPage'
import { AgentListPage } from './pages/agent/AgentListPage'
import { GeneralAgentPage } from './pages/agent/GeneralAgentPage'
import { DocumentParsePage } from './pages/rag/DocumentParsePage'
import { RagChatPage } from './pages/rag/RagChatPage'
import { RagRecordsPage } from './pages/rag/RagRecordsPage'

export default function App() {
  return (
    <Routes>
      <Route element={<AdminLayout />}>
        <Route index element={<Navigate to="/rag/documents" replace />} />
        <Route path="/rag/documents" element={<DocumentParsePage />} />
        <Route path="/rag/records" element={<RagRecordsPage />} />
        <Route path="/rag/chat" element={<RagChatPage />} />
        <Route path="/agents" element={<AgentListPage />} />
        <Route path="/agents/general" element={<GeneralAgentPage />} />
        <Route path="/agents/:agentId" element={<AgentDetailPage />} />
        <Route path="*" element={<Navigate to="/rag/documents" replace />} />
      </Route>
    </Routes>
  )
}
