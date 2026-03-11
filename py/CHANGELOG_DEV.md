# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Added a `--strategy restart` CLI option to `modal deploy` and `app.deploy(strategy="restart")` to restart containers that are older than the latest deployment. The `restart` strategy replaces your old containers faster compared to the default `rolling` strategy. With the `restart` strategy, you may experience higher latency as new containers start up.
- `modal container list` now accepts a `--app-id` to return containers for a specific app.
- `modal serve` will restart running containers when your code gets hot-reloaded.
- Added new SandboxFilesystem namespace with functionality for reading data from and writing data to files in the Sandbox's filesystem.
