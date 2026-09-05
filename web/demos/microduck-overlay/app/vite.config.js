import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: process.env.VITE_BASE || "/",
  plugins: [react()],
  // MuJoCo WASM and onnxruntime-web stay on the CDN (dynamic import with
  // @vite-ignore in game/boot.js), exactly like the pre-Vite app: their
  // .wasm sidecars resolve relative to the CDN URL and never touch the
  // bundle.
  server: {
    port: 5173,
    proxy: {
      "/rdk-api": {
        target: "http://127.0.0.1:8767",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/rdk-api/, ""),
      },
    },
  },
  build: {
    // Keep the JS/CSS bundle out of dist/assets/: the game's static assets
    // (public/assets/) land there and must keep their historical URLs.
    assetsDir: "bundle",
    chunkSizeWarningLimit: 1500,
  },
});
