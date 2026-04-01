# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Adds `modal app rollover my-app --strategy recreate` and `modal app rollover my-app --strategy rolling` CLI commands to rollover your existing deployments. A rollover replaces existing containers with fresh ones built from the same App version — useful for refreshing containers without changing your code.
- Fixes `modal` CLI to be compatiable with older versions of `typer`.
- Added Sandbox readiness probe support in the Python SDK: configure `readiness_probe=` on `modal.Sandbox.create()` using `modal.Probe.with_tcp(...)` or `modal.Probe.with_exec(...)`.
- Added `sandbox.wait_until_ready(timeout=...)` for blocking until a configured Sandbox readiness probe reports ready.
- Added `SandboxFilesystem.remove()` for deleting files and directories from a Sandbox's filesystem.
