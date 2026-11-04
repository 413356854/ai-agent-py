import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发时 `pnpm dev`（5173）把接口与解析结果代理到 FastAPI（8000），
// 生产构建产物由 FastAPI 直接托管在 `/`，因此 base 保持 `/` 即可。
export default defineConfig({
  plugins: [react()],
  server: {
    // 只监听 IPv4 回环：默认的 localhost 在 Node 17+ 可能只绑到 ::1，
    // 部分浏览器/工具走 127.0.0.1 时会连不上；strictPort 保证端口被占时报错而不是偷偷换 5174。
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/parsed': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
  },
})
