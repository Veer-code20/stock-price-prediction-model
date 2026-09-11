import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': process.env.SPPM_API_URL || 'http://127.0.0.1:8001' } },
  test: { environment: 'jsdom', setupFiles: './src/test-setup.js' },
});
