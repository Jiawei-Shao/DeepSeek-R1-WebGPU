import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from 'vite-plugin-static-copy'

// https://vite.dev/config/
export default defineConfig({
  build: {
    target: 'esnext'
  },
  plugins: [tailwindcss(), react(), viteStaticCopy({
    targets: [
      {
        src: 'node_modules/onnxruntime-web/dist/*.jsep.*',
        dest: 'dist'
      }
    ]
  })
],
});
