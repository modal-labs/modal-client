# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Fixes `modal` CLI to be compatiable with older versions of `typer`.
- Added Sandbox readiness probe support in the Python SDK: configure `readiness_probe=` on `modal.Sandbox.create()` using `modal.Probe.with_tcp(...)` or `modal.Probe.with_exec(...)`.
- Added `sandbox.wait_until_ready(timeout=...)` for blocking until a configured Sandbox readiness probe reports ready.
