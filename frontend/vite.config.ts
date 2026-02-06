import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env from project root (parent directory)
  const projectRoot = path.resolve(__dirname, '..')
  const env = loadEnv(mode, projectRoot, '')
  const apiPort = env.VGRAG_API_PORT || '8000'

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@dagrejs/dagre': path.resolve(__dirname, 'node_modules/@dagrejs/dagre/dist/dagre.cjs.js'),
      },
    },
    optimizeDeps: {
      include: ['@dagrejs/dagre', '@dagrejs/graphlib'],
    },
    build: {
      commonjsOptions: {
        include: [/@dagrejs\/dagre/, /@dagrejs\/graphlib/, /node_modules/],
        transformMixedEsModules: true,
      },
    },
    server: {
      proxy: {
        '/api': {
          target: `http://localhost:${apiPort}`,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
