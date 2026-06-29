import { defineConfig } from "vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import babel from "@rolldown/plugin-babel";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset()] }),
    tailwindcss(),
  ],
  server: {
    // Proxy API calls to the local backend so the browser stays same-origin
    // (no CORS) and we avoid the localhost IPv4/IPv6 mismatch: Node connects to
    // 127.0.0.1 explicitly, regardless of how the browser resolves `localhost`.
    // Set VITE_API_BASE_URL=/api in .env.local to route through this.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
