import { defineConfig } from "tsup";
import packageJson from "./package.json" with { type: "json" };

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  define: {
    __MODAL_SDK_VERSION__: JSON.stringify(packageJson.version), // also set in vitest.config.ts
  },
});
