import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * The dev server proxies /api to the real backend rather than mocking it.
 * Every number the UI shows comes from ORION; nothing here fabricates a
 * response. In production the same paths are served by nginx (see
 * docker/nginx.conf), so the app code never needs to know which is in front.
 */
const BACKEND = process.env.ORION_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
