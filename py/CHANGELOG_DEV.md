# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Added `SandboxFilesystem.list_files()` for listing all files in a directory in a Sandbox's filesystem.
- Fixed an issue where images returned by `Sandbox.snapshot_directory` could not be directly passed into `Sandbox.create`.
