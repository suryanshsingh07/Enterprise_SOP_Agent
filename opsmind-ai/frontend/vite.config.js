import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss()
  ],
  base: '/',
  // server: {
  //   proxy: {
  //     '/api': 'http://localhost:5000'
  //   }
  // }
  server: {
    proxy: {
      '/api': {
        target: 'https://opsmind-ai-backend-rinw.onrender.com',
        changeOrigin: true
      },
      '/uploads': {
        target: 'https://opsmind-ai-backend-rinw.onrender.com',
        changeOrigin: true
      }
    }
  }
})
