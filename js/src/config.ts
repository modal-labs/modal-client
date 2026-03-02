import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { parse as parseToml } from "smol-toml";
import { getDefaultClient } from "./client";

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
}

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
  };
  return profile as Profile; // safe to null-cast because of check above
}

/**
 * @deprecated Use `client.environmentName()` instead.
 */
export function environmentName(environment?: string): string {
  return environment || getDefaultClient().profile.environment || "";
}

/**
 * @deprecated Use `client.imageBuilderVersion()` instead.
 */
export function imageBuilderVersion(version?: string): string {
  return version || getDefaultClient().profile.imageBuilderVersion || "2024.10";
}
