import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue()
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    fs: {
      strict: false
    },
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    },
    proxy: {
      '/huggingface': {
        target: 'https://huggingface.co',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/huggingface/, '')
      },
      '/xethub': {
        target: 'https://cas-bridge.xethub.hf.co',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/xethub/, '')
      }
    }
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web', '@huggingface/transformers']
  },
  build: {
    rollupOptions: {
      output: {
        format: 'es'
      }
    }
  }
}) 