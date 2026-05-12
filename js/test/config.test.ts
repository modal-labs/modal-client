import { expect, test, vi } from "vitest";
import { homedir } from "node:os";
import path from "node:path";
import { configFilePath, getProfile } from "../src/config";

const maxThrottleWaitCases = [
  { envVal: "10", expected: 10 },
  { envVal: "0", expected: 0 },
  { envVal: "3600", expected: 3600 },
];

for (const { envVal, expected } of maxThrottleWaitCases) {
  test(`GetProfile_MaxThrottleWaitParsing/${envVal}`, () => {
    vi.stubEnv("MODAL_MAX_THROTTLE_WAIT", envVal);
    const profile = getProfile();
    expect(profile.maxThrottleWaitSecs).toBe(expected);
    vi.unstubAllEnvs();
  });
}

test("GetProfile_MaxThrottleWaitInvalidValue", () => {
  vi.stubEnv("MODAL_MAX_THROTTLE_WAIT", "not-a-number");
  const profile = getProfile();
  expect(profile.maxThrottleWaitSecs).toBeUndefined();
  vi.unstubAllEnvs();
});

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
