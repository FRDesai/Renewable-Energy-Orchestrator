import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The console talks to the decision service over the same origin, so the WebSocket and
// the REST calls share it and no CORS pre-flight is needed. `ORCH_API` moves the target
// when :8000 is taken.
const target = process.env["ORCH_API"] ?? "http://127.0.0.1:8000";
const wsTarget = target.replace(/^http/, "ws");

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env["ORCH_PORT"] ?? 5173),
    proxy: {
      "/api": { target, changeOrigin: true },
      "/ws": { target: wsTarget, ws: true },
    },
  },
  preview: {
    port: Number(process.env["ORCH_PORT"] ?? 5173),
    proxy: {
      "/api": { target, changeOrigin: true },
      "/ws": { target: wsTarget, ws: true },
    },
  },
});
