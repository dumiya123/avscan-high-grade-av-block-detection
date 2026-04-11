import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
    plugins: [react()],
    build: {
        rollupOptions: {
            output: {
                manualChunks: {
                    'vendor': ['react', 'react-dom', 'framer-motion', 'lucide-react', 'clsx', 'tailwind-merge'],
                    'plotly-vendor': ['plotly.js-dist-min', 'react-plotly.js']
                }
            }
        },
        chunkSizeWarningLimit: 1000
    }
})
