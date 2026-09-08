import { expect, test, vi } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { tmpdir } from "node:os";
import path from "node:path";
import { configFilePath, getProfile } from "../src/config";
import { DEFAULT_SANDBOX_CHANNEL_IDLE_TIMEOUT_MS } from "../src/config";

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

const sandboxV2Cases = [
  { envVal: undefined, expected: false },
  { envVal: "", expected: false },
  { envVal: "0", expected: false },
  { envVal: "false", expected: false },
  { envVal: "False", expected: false },
  { envVal: "1", expected: true },
  { envVal: "true", expected: true },
  { envVal: "yes", expected: true },
];

for (const { envVal, expected } of sandboxV2Cases) {
  test(`GetProfile_SandboxV2Parsing/${JSON.stringify(envVal)}`, () => {
    vi.stubEnv("MODAL_SANDBOX_V2", envVal);
    const profile = getProfile();
    expect(profile.sandboxV2).toBe(expected);
    vi.unstubAllEnvs();
  });
}

test("GetProfile_SandboxV2FromConfig", async () => {
  const configDir = mkdtempSync(path.join(tmpdir(), "modal-js-config-"));
  const configPath = path.join(configDir, ".modal.toml");
  writeFileSync(
    configPath,
    `
[v2-profile]
sandbox_v2 = true
image_builder_version = "2024.10"
`,
  );
  vi.stubEnv("MODAL_CONFIG_PATH", configPath);
  vi.stubEnv("MODAL_SANDBOX_V2", undefined);
  vi.stubEnv("MODAL_IMAGE_BUILDER_VERSION", undefined);
  vi.resetModules();

  try {
    const { getProfile: getProfileFromConfig } = await import("../src/config");
    const profile = getProfileFromConfig("v2-profile");
    expect(profile.sandboxV2).toBe(true);
    expect(profile.imageBuilderVersion).toBe("2024.10");
  } finally {
    vi.unstubAllEnvs();
    vi.resetModules();
    rmSync(configDir, { recursive: true });
  }
});

test("GetProfile_SandboxV2QuotedValuesInConfig", async () => {
  const configDir = mkdtempSync(path.join(tmpdir(), "modal-js-config-"));
  const configPath = path.join(configDir, ".modal.toml");
  writeFileSync(
    configPath,
    `
[quoted-false]
sandbox_v2 = "false"

[quoted-true]
sandbox_v2 = "true"
`,
  );
  vi.stubEnv("MODAL_CONFIG_PATH", configPath);
  vi.stubEnv("MODAL_SANDBOX_V2", undefined);
  vi.resetModules();

  try {
    const { getProfile: getProfileFromConfig } = await import("../src/config");
    expect(getProfileFromConfig("quoted-false").sandboxV2).toBe(false);
    expect(getProfileFromConfig("quoted-true").sandboxV2).toBe(true);
  } finally {
    vi.unstubAllEnvs();
    vi.resetModules();
    rmSync(configDir, { recursive: true });
  }
});

test("GetProfile_SandboxV2EnvOverridesConfig", async () => {
  const configDir = mkdtempSync(path.join(tmpdir(), "modal-js-config-"));
  const configPath = path.join(configDir, ".modal.toml");
  writeFileSync(
    configPath,
    `
[v2-profile]
sandbox_v2 = true
`,
  );
  vi.stubEnv("MODAL_CONFIG_PATH", configPath);
  vi.stubEnv("MODAL_SANDBOX_V2", "0");
  vi.resetModules();

  try {
    const { getProfile: getProfileFromConfig } = await import("../src/config");
    const profile = getProfileFromConfig("v2-profile");
    expect(profile.sandboxV2).toBe(false);
  } finally {
    vi.unstubAllEnvs();
    vi.resetModules();
    rmSync(configDir, { recursive: true });
  }
});

test("GetProfile_OAuthCredentials", () => {
  vi.stubEnv("MODAL_OAUTH_REFRESH_TOKEN", "refresh-token");
  vi.stubEnv("MODAL_OAUTH_CLIENT_ID", "oc-client-id");
  vi.stubEnv("MODAL_OAUTH_CLIENT_SECRET", "ov-client-secret");

  const profile = getProfile();
  expect(profile.oauthRefreshToken).toBe("refresh-token");
  expect(profile.oauthClientId).toBe("oc-client-id");
  expect(profile.oauthClientSecret).toBe("ov-client-secret");
  vi.unstubAllEnvs();
});

test("GetProfile_OAuthCredentialsFromConfig", async () => {
  const configDir = mkdtempSync(path.join(tmpdir(), "modal-js-config-"));
  const configPath = path.join(configDir, ".modal.toml");
  writeFileSync(
    configPath,
    `
[oauth-profile]
oauth_refresh_token = "refresh-token"
oauth_client_id = "oc-client-id"
oauth_client_secret = "ov-client-secret"
`,
  );
  vi.stubEnv("MODAL_CONFIG_PATH", configPath);
  vi.stubEnv("MODAL_OAUTH_REFRESH_TOKEN", undefined);
  vi.stubEnv("MODAL_OAUTH_CLIENT_ID", undefined);
  vi.stubEnv("MODAL_OAUTH_CLIENT_SECRET", undefined);
  vi.resetModules();

  try {
    const { getProfile: getProfileFromConfig } = await import("../src/config");
    const profile = getProfileFromConfig("oauth-profile");
    expect(profile.oauthRefreshToken).toBe("refresh-token");
    expect(profile.oauthClientId).toBe("oc-client-id");
    expect(profile.oauthClientSecret).toBe("ov-client-secret");
  } finally {
    vi.unstubAllEnvs();
    vi.resetModules();
    rmSync(configDir, { recursive: true });
  }
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
}

// Values that are timeouts still get through, zero included: it turns the
// release off rather than releasing at once.
test("GetProfile_IdleTimeoutAcceptsSeconds", () => {
  vi.stubEnv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", "0");
  const profile = getProfile();
  expect(profile.sandboxChannelIdleTimeoutMs).toBe(0);
  vi.unstubAllEnvs();
});

// A positive timeout too short to round to a millisecond must not read as zero,
// which is how the release is turned off - that would invert what was asked for.
test("GetProfile_IdleTimeoutKeepsAShortTimeoutPositive", () => {
  vi.stubEnv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", "0.0004");
  const profile = getProfile();
  expect(profile.sandboxChannelIdleTimeoutMs).toBeGreaterThan(0);
  vi.unstubAllEnvs();
});
