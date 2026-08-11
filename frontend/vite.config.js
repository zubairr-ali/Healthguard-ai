import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// HealthGuard AI — frontend build config.
// The dev proxy forwards /api/* to the local FastAPI backend (uvicorn on :8000)
// so the frontend can call relative paths in both dev and production.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
