import { defineConfig } from "vitest/config";
import path from "node:path";

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
});
