import { expect, test, vi } from "vitest";
import { homedir } from "node:os";
import path from "node:path";
import { configFilePath, getProfile } from "../src/config";
import {
  DEFAULT_SANDBOX_CHANNEL_IDLE_TIMEOUT_MS,
  DEFAULT_SANDBOX_STREAM_IDLE_TIMEOUT_MS,
} from "../src/config";

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

// A value that parses as a number is not necessarily a timeout. Infinity is
// not one, and setTimeout treats anything past 2^31-1 ms as zero, so a value
// that large would fire at once rather than never.
const notATimeout = ["Infinity", "-Infinity", "NaN", "1e30", "-1", "nonsense"];

for (const value of notATimeout) {
  test(`GetProfile_ChannelIdleTimeoutRejects/${value}`, () => {
    vi.stubEnv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", value);
    const profile = getProfile();
    expect(profile.sandboxChannelIdleTimeoutMs).toBe(
      DEFAULT_SANDBOX_CHANNEL_IDLE_TIMEOUT_MS,
    );
    vi.unstubAllEnvs();
  });

  test(`GetProfile_StreamIdleTimeoutRejects/${value}`, () => {
    vi.stubEnv("MODAL_SANDBOX_STREAM_IDLE_TIMEOUT", value);
    const profile = getProfile();
    expect(profile.sandboxStreamIdleTimeoutMs).toBe(
      DEFAULT_SANDBOX_STREAM_IDLE_TIMEOUT_MS,
    );
    vi.unstubAllEnvs();
  });
}

// Values that are timeouts still get through, zero included: it turns the
// release off rather than releasing at once.
test("GetProfile_IdleTimeoutAcceptsSeconds", () => {
  vi.stubEnv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", "0");
  vi.stubEnv("MODAL_SANDBOX_STREAM_IDLE_TIMEOUT", "2.5");
  const profile = getProfile();
  expect(profile.sandboxChannelIdleTimeoutMs).toBe(0);
  expect(profile.sandboxStreamIdleTimeoutMs).toBe(2500);
  vi.unstubAllEnvs();
});

// A positive timeout too short to round to a millisecond must not read as zero,
// which is how the release is turned off - that would invert what was asked for.
test("GetProfile_IdleTimeoutKeepsAShortTimeoutPositive", () => {
  vi.stubEnv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", "0.0004");
  vi.stubEnv("MODAL_SANDBOX_STREAM_IDLE_TIMEOUT", "0.0004");
  const profile = getProfile();
  expect(profile.sandboxChannelIdleTimeoutMs).toBeGreaterThan(0);
  expect(profile.sandboxStreamIdleTimeoutMs).toBeGreaterThan(0);
  vi.unstubAllEnvs();
});
