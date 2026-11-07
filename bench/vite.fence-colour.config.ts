// Production build of the fence-colour measurement page (copied into each tree under test).
import path from "node:path";
import { mergeConfig } from "vite";
import base from "./vite.config";

export default mergeConfig(base, {
  build: {
    emptyOutDir: true,
    minify: process.env.FC_MINIFY !== "0",
    outDir: process.env.FENCE_COLOUR_OUT as string,
    rollupOptions: {
      input: { fence: path.resolve(__dirname, "smoke-fence-colour.html") },
    },
  },
});
