import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const FRONTEND = path.resolve(__dirname, "..");

export default defineConfig({
  root: FRONTEND,
  plugins: [react(), tailwindcss()],
  css: { postcss: { plugins: [] } },
  resolve: { alias: { "@": path.resolve(FRONTEND, "src") } },
  build: {
    outDir: path.resolve(__dirname, "dist"),
    emptyOutDir: true,
    rollupOptions: { input: path.resolve(__dirname, "index.html") },
  },
});
