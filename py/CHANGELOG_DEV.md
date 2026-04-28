# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Added `SandboxFilesystem.list_files()` for listing all entries (and their metadata) in a directory in a Sandbox's filesystem.
- Added `SandboxFilesystem.stat()` for summarizing the metadata of a given file/symlink/directory a Sandbox's filesystem.
- Improved reliability and latency of `modal container exec`, `modal shell`, and `modal cluster shell`.
- Fixed an issue where images returned by `Sandbox.snapshot_directory` could not be directly passed into `Sandbox.create`.
- It's now possible to pass a custom App name for ephemeral Apps using the `--name` option in `modal run` / `modal serve` or by setting `name=` in `App.run()`.
