import { warn } from "node:console";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { parse as parseToml } from "smol-toml";

/** Raw representation of the .modal.toml file. */
interface Config {
  [profile: string]: {
    server_url?: string;
    token_id?: string;
    token_secret?: string;
    environment?: string;
    imageBuilderVersion?: string;
    loglevel?: string;
    active?: boolean;
  };
}

/** Resolved configuration object from `Config` and environment variables. */
export interface Profile {
  serverUrl: string;
  tokenId?: string;
  tokenSecret?: string;
  environment?: string;
  imageBuilderVersion?: string;
  logLevel?: string;
  /** Parsed from MODAL_MAX_THROTTLE_WAIT. null means unlimited. */
  maxThrottleWaitSecs?: number;
  /**
   * How long a Sandbox connection may sit idle before the client gives it up.
   * The Sandbox stays usable: the next operation reconnects. Zero keeps
   * connections open until the client closes.
   */
  sandboxChannelIdleTimeoutMs: number;
  /**
   * How long a caller may sit on a chunk of a Sandbox's output before the stream
   * stops counting as in use. Only once a reader has gone quiet for this long
   * does the idle timeout above start running, so a Sandbox read once and then
   * forgotten gives its connection up after the two together. Zero stops
   * counting as soon as a caller stops reading.
   */
}

/**
 * How long a Sandbox connection may sit idle before it is released, unless
 * MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT says otherwise.
 */
export const DEFAULT_SANDBOX_CHANNEL_IDLE_TIMEOUT_MS = 30_000;

/**
 * The largest delay setTimeout honours. Past this it wraps to zero and the
 * timer fires immediately, so a timeout this long is refused rather than
 * silently inverted.
 */
const MAX_TIMEOUT_MS = 2_147_483_647;

export function isLocalhost(profile: Profile): boolean {
  const url = new URL(profile.serverUrl);
  const hostname = url.hostname;
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "::1" ||
    hostname === "172.21.0.1"
  );
}

export function configFilePath(): string {
  const configPath = process.env["MODAL_CONFIG_PATH"];
  if (configPath && configPath !== "") {
    return configPath;
  }
  return path.join(homedir(), ".modal.toml");
}

function readConfigFile(): Config {
  try {
    const configPath = configFilePath();
    const configContent = readFileSync(configPath, {
      encoding: "utf-8",
    });
    return parseToml(configContent) as Config;
  } catch (err: any) {
    if (err.code === "ENOENT") {
      return {} as Config;
    }
    // Ignore failure to read or parse .modal.toml
    // throw new Error(`Failed to read or parse .modal.toml: ${err.message}`);
    return {} as Config;
  }
}

// Synchronous on startup to avoid top-level await in CJS output.
//
// Any performance impact is minor because the .modal.toml file is small and
// only read once. This is comparable to how OpenSSL certificates can be probed
// synchronously, for instance.
const config: Config = readConfigFile();

export function getProfile(profileName?: string): Profile {
  if (!profileName) {
    for (const [name, profileData] of Object.entries(config)) {
      if (profileData.active) {
        profileName = name;
        break;
      }
    }
  }
  const profileData =
    profileName && Object.hasOwn(config, profileName)
      ? config[profileName]
      : {};

  const profile: Partial<Profile> = {
    serverUrl:
      process.env["MODAL_SERVER_URL"] ||
      profileData.server_url ||
      "https://api.modal.com:443",
    tokenId: process.env["MODAL_TOKEN_ID"] || profileData.token_id,
    tokenSecret: process.env["MODAL_TOKEN_SECRET"] || profileData.token_secret,
    environment: process.env["MODAL_ENVIRONMENT"] || profileData.environment,
    imageBuilderVersion:
      process.env["MODAL_IMAGE_BUILDER_VERSION"] ||
      profileData.imageBuilderVersion,
    logLevel: process.env["MODAL_LOGLEVEL"] || profileData.loglevel,
    maxThrottleWaitSecs: (() => {
      const val = process.env["MODAL_MAX_THROTTLE_WAIT"];
      if (!val) return undefined;
      const parsed = parseInt(val, 10);
      if (isNaN(parsed) || parsed < 0) {
        // We use `warn` here because Modal's logger is constructed after the profile is constructed
        warn(
          `MODAL_MAX_THROTTLE_WAIT="${val}" is not a valid non-negative integer; ignoring.`,
        );
        return undefined;
      }
      return parsed;
    })(),
    sandboxChannelIdleTimeoutMs: (() => {
      const val = process.env["MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT"];
      if (!val) return DEFAULT_SANDBOX_CHANNEL_IDLE_TIMEOUT_MS;
      const parsed = Number(val);
      // Infinity parses as a number but is not a timeout, and setTimeout treats
      // anything past MAX_TIMEOUT_MS as zero - firing at once rather than never.
      if (
        !Number.isFinite(parsed) ||
        parsed < 0 ||
        parsed * 1000 > MAX_TIMEOUT_MS
      ) {
        // We use `warn` here because Modal's logger is constructed after the profile is constructed
        warn(
          `MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT="${val}" is not a valid non-negative number of seconds; ignoring.`,
        );
        return DEFAULT_SANDBOX_CHANNEL_IDLE_TIMEOUT_MS;
      }
      const ms = Math.round(parsed * 1000);
      // Zero is how a caller turns the release off, so a timeout too short to
      // round to a millisecond becomes the shortest one there is rather than
      // none at all.
      return parsed > 0 && ms === 0 ? 1 : ms;
    })(),
  };
  return profile as Profile; // safe to null-cast because of check above
}
