import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { onTestFinished } from "vitest";

import type { Sandbox } from "../src/sandbox";
import type { FileWatchEvent, FileWatchEventType } from "../src/sandbox_fs";

export async function writeRemoteFile(
  sb: Sandbox,
  path: string,
  data: Uint8Array,
): Promise<void> {
  const p = await sb.exec(
    ["sh", "-c", 'mkdir -p "$(dirname "$1")" && cat > "$1"', "--", path],
    { mode: "binary" },
  );
  const writer = p.stdin.getWriter();
  try {
    await writer.write(data);
    await writer.close();
  } finally {
    writer.releaseLock();
  }
  const rc = await p.wait();
  if (rc !== 0)
    throw new Error(`writeRemoteFile failed for ${path} (rc=${rc})`);
}

export async function readRemoteFile(
  sb: Sandbox,
  path: string,
): Promise<Uint8Array> {
  const p = await sb.exec(["cat", path], { mode: "binary" });
  const [data, rc] = await Promise.all([p.stdout.readBytes(), p.wait()]);
  if (rc !== 0) throw new Error(`readRemoteFile failed for ${path} (rc=${rc})`);
  return data;
}

/** Generate deterministic pseudorandom bytes. */
export function randomBytes(size: number, seed: number): Uint8Array {
  let state = seed >>> 0;
  const buf = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    buf[i] = state & 0xff;
  }
  return buf;
}

export async function tmpPath(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "modal-fs-e2e-"));
  onTestFinished(() => rm(dir, { recursive: true, force: true }));
  return dir;
}

export async function mkdirRemote(sb: Sandbox, path: string): Promise<void> {
  const p = await sb.exec(["mkdir", "-p", path]);
  const rc = await p.wait();
  if (rc !== 0) throw new Error(`mkdirRemote failed for ${path} (rc=${rc})`);
}

export async function pathExists(sb: Sandbox, path: string): Promise<boolean> {
  const p = await sb.exec(["test", "-e", path]);
  return (await p.wait()) === 0;
}

export async function isDirRemote(sb: Sandbox, path: string): Promise<boolean> {
  const p = await sb.exec(["test", "-d", path]);
  return (await p.wait()) === 0;
}

export async function createSparseFile(
  sb: Sandbox,
  path: string,
  sizeBytes: number,
): Promise<void> {
  const p = await sb.exec(["truncate", "-s", String(sizeBytes), path]);
  const rc = await p.wait();
  if (rc !== 0)
    throw new Error(`createSparseFile failed for ${path} (rc=${rc})`);
}

export async function symlinkRemote(
  sb: Sandbox,
  target: string,
  linkPath: string,
): Promise<void> {
  const p = await sb.exec(["ln", "-s", target, linkPath]);
  const rc = await p.wait();
  if (rc !== 0)
    throw new Error(
      `symlinkRemote failed: ${target} -> ${linkPath} (rc=${rc})`,
    );
}

/** Runs `stat -c '%a %U %G %Y %f' path` in the sandbox for ground-truth metadata. */
export async function statRemoteFile(
  sb: Sandbox,
  path: string,
): Promise<{
  permissions: string;
  owner: string;
  group: string;
  mtime: number;
  mode: number;
}> {
  const p = await sb.exec(["stat", "-c", "%a %U %G %Y %f", path]);
  const [stdout, rc] = await Promise.all([p.stdout.readText(), p.wait()]);
  if (rc !== 0) throw new Error(`statRemoteFile failed for ${path}`);
  const [perms, owner, group, mtime, rawMode] = stdout.trim().split(" ");
  return {
    permissions: perms.padStart(4, "0"),
    owner,
    group,
    mtime: parseFloat(mtime),
    mode: parseInt(rawMode, 16),
  };
}

/**
 * Start a background exec loop in the sandbox (`triggerShellCmd` runs every
 * ~100 ms for `triggerSecs` seconds), run the watch for `opts.timeoutMs`
 * milliseconds, then return all collected events.
 */
export async function watchWithSandboxTrigger(
  sb: Sandbox,
  path: string,
  triggerShellCmd: string,
  opts: {
    timeoutMs?: number;
    filter?: FileWatchEventType[];
    recursive?: boolean;
  } = {},
): Promise<FileWatchEvent[]> {
  const { timeoutMs = 2000 } = opts;
  const triggerSecs = Math.ceil(timeoutMs / 1000) + 2;
  const bg = await sb.exec([
    "sh",
    "-c",
    `end=$(($(date +%s)+${triggerSecs})); while [ "$(date +%s)" -lt "$end" ]; do ${triggerShellCmd}; sleep 0.1; done`,
  ]);
  const events: FileWatchEvent[] = [];
  try {
    for await (const event of sb.filesystem.watch(path, opts)) {
      events.push(event);
    }
  } finally {
    await bg.wait();
  }
  return events;
}
