import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
    plugins: [react()],
    esbuild: {
        drop: ['console', 'debugger'],
    },
    build: {
        rollupOptions: {
            output: {
                manualChunks: {
                    'react-vendor': ['react', 'react-dom'],
                    'lucide': ['lucide-react'],
                    'plotly-vendor': ['plotly.js-dist-min', 'react-plotly.js']
                }
            }
        },
        chunkSizeWarningLimit: 1500
    }
})
