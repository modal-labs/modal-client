# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Adds `modal app rollover my-app --strategy recreate` and `modal app rollover my-app --strategy rolling` CLI commands to rollover your existing deployments. A rollover replaces existing containers with fresh ones built from the same App version — useful for refreshing containers without changing your code.
- The new `modal bootstrap` CLI command will fetch deployable starter code for common AI applications (e.g., text generation, text-to-image, speech-to-text). This is an experiment: try it out and give us feedback!
- The `modal app stop` and `modal container stop` CLIs now require a confirmation step or a `--yes` flag.
- Added `SandboxFilesystem.remove()` for deleting files and directories from a Sandbox's filesystem.
- Added `SandboxFilesystem.make_directory()` for creating directories in a Sandbox's filesystem.
- Added `build_args` parameter to `Image.dockerfile_commands()`, matching existing support in `Image.from_dockerfile()`.
