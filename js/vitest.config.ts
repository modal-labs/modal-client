import { defineConfig } from "vitest/config";
import path from "node:path";
import packageJson from "./package.json" with { type: "json" };

export default defineConfig({
  test: {
    maxConcurrency: 10,
    slowTestThreshold: 5_000,
    testTimeout: 20_000,
    reporters: ["verbose"],
  },
  resolve: {
    alias: {
      modal: path.resolve(__dirname, "./src/index.ts"),
    },
  },
  define: {
    __MODAL_SDK_VERSION__: JSON.stringify(packageJson.version), // also set in tsup.config.ts
  },
});
