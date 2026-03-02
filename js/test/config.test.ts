import { expect, test, vi } from "vitest";
import { homedir } from "node:os";
import path from "node:path";
import { configFilePath } from "../src/config";

test("GetConfigPath_WithEnvVar", () => {
  const customPath = "/custom/path/to/config.toml";
  vi.stubEnv("MODAL_CONFIG_PATH", customPath);

  const result = configFilePath();
  expect(result).toBe(customPath);

  vi.unstubAllEnvs();
});

test("GetConfigPath_WithoutEnvVar", () => {
  vi.stubEnv("MODAL_CONFIG_PATH", undefined);

  const result = configFilePath();
  const expectedPath = path.join(homedir(), ".modal.toml");
  expect(result).toBe(expectedPath);

  vi.unstubAllEnvs();
});
