# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- Sandbox Sidecars can now snapshot their filesystems into reusable Images with `sidecar.snapshot_filesystem()`.
- Added the [`Sandbox.logs`](/docs/sdk/py/latest/Sandbox#logs) namespace to retrieve Sandbox entrypoint logs directly from the SDK. The namespace has two different methods, allowing you to `fetch()` logs from a specific date/time range, or `tail()` the most recent logs.
- Added support for setting the default member Role when creating Restricted Environments through the Python SDK and CLI.
- The `modal` CLI now accepts a global `--profile` option for simpler ad hoc profile selection.
- Fixed gRPC channels attempting to reuse connections whose underlying transport is closing.
- `modal environment roles list --exclude-default` and
  `Environment.roles.list(exclude_default=True)` list only users and service users who have been
  directly assigned a role for the Environment.
- Added implicit OAuth refresh token authentication through the `MODAL_OAUTH_REFRESH_TOKEN`, `MODAL_OAUTH_CLIENT_ID`, and `MODAL_OAUTH_CLIENT_SECRET` environment variables, with [`modal.Client.from_oauth_credentials()`](/docs/sdk/py/latest/Client#from_oauth_credentials) available for explicitly constructed clients.
- Added deprecation warnings for the following methods:
  - `_Object.deps`, `_Object.is_hydrated`, `_Object.local_uuid`
  - `_Function.from_local`, `_Function.get_build_def`, `_Function.get_raw_f`, `_Function.info`, `_Function.is_generator`, `_Function.spec`, `_Function.stub`, `_Function.tag`
  - `_App.image`, `_App.is_interactive`, `_App.registered_classes`, `_App.registered_entrypoints`, `_App.registered_functions`, `_App.registered_web_endpoints`, `_App.set_description`
  - `_Cls.validate_construction_mechanism`, `_Cls.from_local`
- Added `modal app info` CLI command that displays the constituent functions and servers of an app as well as its deployment lifecycle information.
- Added `App.info` method that displays the constituent functions and servers of an app as well as its deployment lifecycle information.

## JS

- Sandbox Sidecars can now snapshot their filesystems into reusable Images with `SidecarContainer.snapshotFilesystem`.
- A client now releases its connection to a Sandbox once that Sandbox has been idle for 30 seconds, freeing associated resources similar to an explicit `.detach()` call. A subsequent sandbox operation reconnects so the Sandbox stays usable. `Sandbox.terminate()` no longer detaches from the Sandbox and instead relies on the new idle timeout or an explicit `.detach()` call to release resources.
- A Sandbox's stdout/stderr, and the same streams on `ContainerProcess` (as returned by `sandbox.exec()`), now defer fetching output until `.stdout`/`.stderr` is first read. This makes the JS SDK consistent with the Go/Python SDKs in this regard.
- A Sandbox's `stdout`/`stderr`, and those of a `ContainerProcess`, no longer read ahead of the caller: output is fetched by the read that asks for it, with no background prefetching/buffering. Pipe them through a `TransformStream` or similar to add your own read-ahead if needed.
- Fixed an issue where `readText()` called on a binary-mode Sandbox stream could sometimes return incorrect output data.

## Go

- Sandbox Sidecars can now snapshot their filesystems into reusable Images with `SidecarContainer.SnapshotFilesystem`.
- A client now releases its connection to a Sandbox once that Sandbox has been idle for 30 seconds, freeing associated resources similar to an explicit `Detach()` call. A subsequent sandbox operation reconnects so the Sandbox stays usable. `Sandbox.Terminate()` no longer detaches from the Sandbox and instead relies on the new idle timeout or an explicit `Detach()` call to release resources.
- A Sandbox's `Stdout`/`Stderr`, and those of a `ContainerProcess`, no longer read ahead of the caller: output is fetched by the read that asks for it, with no background goroutine and no buffer in between. Wrap them in a `bufio.Reader` for read-ahead.
