# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Added a `tags` parameter to `Sandbox.create` so tags can be set at creation time instead of via a separate `Sandbox.set_tags` call.
- Added `SandboxFilesystem.list_files()` for listing all entries (and their metadata) in a directory in a Sandbox's filesystem.
- Added `SandboxFilesystem.stat()` for summarizing the metadata of a given file/symlink/directory a Sandbox's filesystem.
- Added a new `modal.Environment` object for environment management, including RBAC configuration.
- Improved reliability and latency of `modal container exec`, `modal shell`, and `modal cluster shell`.
- Added an `Image.pipe` method to let you define reusable Image recipes that compose well with the fluent Image builder interface.
- Fixed an issue where images returned by `Sandbox.snapshot_directory` could not be directly passed into `Sandbox.create`.
- It's now possible to pass a custom App name for ephemeral Apps using the `--name` option in `modal run` / `modal serve` or by setting `name=` in `App.run()`.
- Enabled --chmod and --chown flags for COPY commands, within modal.Image.from_dockerfile
- Added an `inbound_cidr_allowlist` parameter to `Sandbox.create()` to restrict which source IPs can connect inbound to a sandbox's tunnels and connection tokens.
- Renamed the `cidr_allowlist` parameter in `Sandbox.create()` to `outbound_cidr_allowlist` to distinguish from the inbound allowlist. The old name is deprecated and will be removed in a future release.
