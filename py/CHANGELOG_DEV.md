# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- The `modal app logs` and `modal container logs` CLIs will no longer follow logs by default. It's now necessary to opt-into following by passing `--follow` or `-f`.
- Added a `--strategy restart` CLI option to `modal deploy` and `app.deploy(strategy="restart")` to restart containers that are older than the latest deployment. The `restart` strategy replaces your old containers faster compared to the default `rolling` strategy. With the `restart` strategy, you may experience higher latency as new containers start up.
- `modal container list` now accepts a `--app-id` to return containers for a specific app.
- `modal serve` will restart running containers when your code gets hot-reloaded.
- Added new SandboxFilesystem namespace with functionality for reading data from and writing data to files in the Sandbox's filesystem.
- `Sandbox.create` now accepts an `include_oidc_identity_token` parameter. When set to `True`, a `MODAL_IDENTITY_TOKEN` environment variable is injected into the sandbox, enabling OIDC-based authentication (e.g., for AWS federation). See the [OIDC integration guide](https://modal.com/docs/guide/oidc-integration) for more details.
