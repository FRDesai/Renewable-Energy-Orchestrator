import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: false,
    setupFiles: ["./src/test/setup.ts"],
    // A base URL so the client's relative paths are valid URLs under jsdom, exactly as
    // they are in the browser where the console is served from the same origin.
    environmentOptions: { jsdom: { url: "http://localhost:5173/" } },
  },
});
