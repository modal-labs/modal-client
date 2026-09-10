# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- Removed the deprecated legacy Sandbox filesystem API: `Sandbox.open()`, `Sandbox.ls()`, `Sandbox.mkdir()`, `Sandbox.rm()`, `Sandbox.watch()`, the `modal.file_io.FileIO` type, and `modal.exception.FilesystemExecutionError`. Use the [`Sandbox.filesystem`](/docs/sdk/py/latest/Sandbox#filesystem) APIs instead.
- [`FunctionCall.get_call_graph()`](/docs/sdk/py/latest/FunctionCall#get_call_graph) now returns up to 5000
  nodes instead of 100.
- Volume file downloads no longer transfer the trailing zero bytes of sparse file blocks, which the client now fills in locally.
- Sandbox Sidecars can now snapshot their filesystems into reusable Images with `sidecar.snapshot_filesystem()`.
- Added `mount_image()`, `unmount_image()`, and `snapshot_directory()` to experimental Sandbox Sidecar containers. Directory snapshot Images can be used anywhere an Image is accepted, including as mounts and as container filesystems.
- A client's connection to a Sandbox is now dropped automatically after a period of inactivity, and reconnected if the Sandbox is used again. This makes `Sandbox.detach()` optional. `Sandbox.detach()` does not interrupt or wait for running concurrent operations on the Sandbox; connections are promptly closed once those operations complete. Use `Sandbox.from_id()` to obtain a new handle if you need to continue interacting with the Sandbox.
- Added the [`Sandbox.logs`](/docs/sdk/py/latest/Sandbox#logs) namespace to retrieve Sandbox entrypoint logs directly from the SDK. The namespace has two different methods, allowing you to `fetch()` logs from a specific date/time range, or `tail()` the most recent logs.
- Added support for setting the default member Role when creating Restricted Environments through the Python SDK and CLI.
- The `modal` CLI now accepts a global `--profile` option for simpler ad hoc profile selection.
- Fixed gRPC channels attempting to reuse connections whose underlying transport is closing.
- Sandbox and `ContainerProcess` output streams now reset their transient-error retry budget when output is received, so the budget limits consecutive failures rather than failures over the stream's lifetime.
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
- Added `Function.info` method that displays static information about a particular Function
- Added `Server.info` method that displays static information about a particular Server
- Added `modal function stats` to inspect performance metrics for a deployed Function over a selected time window.

## JS

- It's now possible to opt into a [more performant Sandbox backend](/blog/scaling-to-1-million-concurrent-sandboxes-in-seconds) by setting the `MODAL_SANDBOX_V2=1` environment variable or the `sandbox_v2 = true` profile key in `.modal.toml`: `sandboxes.create`, `sandboxes.fromName`, and `sandboxes.list` then use the new backend without any code changes. This will become the default behavior in an upcoming release; setting the flag lets you opt in early.
- Fixed the `image_builder_version` profile key in `.modal.toml` not being read: the JS SDK previously looked for it under a different name than the one the Python SDK writes.
- Added support for authenticating with OAuth refresh tokens and client credentials.
- Sandbox Sidecars can now snapshot their filesystems into reusable Images with `SidecarContainer.snapshotFilesystem`.
- A client now releases its dedicated connection to a Sandbox once that connection has been idle for 30 seconds, freeing associated resources. V2 Sandbox output and `ContainerProcess` output use the same idle timeout. Subsequent operations and reads reconnect after an idle release. `Sandbox.detach()` prevents new operations on this connection and releases it as soon as active calls and output pulls finish, including their retries; detach does not wait for or cancel them. Existing output streams may still deliver buffered data. Accessing V2 `stdout` or `stderr` for the first time after detach raises `ClientClosedError`. V1 Sandbox output streams remain readable after detach; unread streams are released after the same idle timeout and resume from their last log entry on the next read. A timeout of zero disables idle cleanup. `Sandbox.terminate()` no longer detaches and instead relies on idle cleanup or an explicit `.detach()` call to release the dedicated connection.
- Added `mountImage()`, `unmountImage()`, and `snapshotDirectory()` to experimental Sandbox Sidecar containers. Directory snapshot Images can be used anywhere an Image is accepted, including as mounts and as container filesystems.
- A Sandbox's stdout/stderr, and the same streams on `ContainerProcess` (as returned by `sandbox.exec()`), now defer fetching output until `.stdout`/`.stderr` is first read. This makes the JS SDK consistent with the Go/Python SDKs in this regard.
- A Sandbox's `stdout`/`stderr`, and those of a `ContainerProcess`, no longer read ahead of the caller: output is fetched by the read that asks for it, with no background prefetching/buffering. Pipe them through a `TransformStream` or similar to add your own read-ahead if needed.
- Fixed an issue where `readText()` called on a binary-mode Sandbox stream could sometimes return incorrect output data.
- Fixed an `AbortSignal` passed to a streaming API being dropped before it reached the call, so cancelling one now ends the call rather than only ending the caller's iteration.

## Go

- It's now possible to opt into a [more performant Sandbox backend](/blog/scaling-to-1-million-concurrent-sandboxes-in-seconds) by setting the `MODAL_SANDBOX_V2=1` environment variable or the `sandbox_v2 = true` profile key in `.modal.toml`: `Sandboxes.Create`, `Sandboxes.FromName`, and `Sandboxes.List` then use the new backend without any code changes. This will become the default behavior in an upcoming release; setting the flag lets you opt in early.
- Added support for authenticating with OAuth refresh tokens and client credentials.
- Added support for OAuth private-key JWT authentication via `MODAL_OAUTH_JWT_KEY` / `OAuthCredentialsParams.JWTKey`. Provide exactly one of client secret or JWT key.
- Sandbox Sidecars can now snapshot their filesystems into reusable Images with `SidecarContainer.SnapshotFilesystem`.
- A client now releases its dedicated connection to a Sandbox once that connection has been idle for 30 seconds, freeing associated resources. V2 Sandbox output and `ContainerProcess` output use the same idle timeout. Subsequent operations and reads reconnect after an idle release. `Sandbox.Detach()` prevents new operations on this connection and releases it as soon as active calls and output reads finish, including their retries; detach does not wait for or cancel them. Further V2 Sandbox output reads return `ClientClosedError`. V1 Sandbox output streams remain readable after detach; unread streams are released after the same idle timeout and resume from their last log entry on the next read. A timeout of zero disables idle cleanup. These log streams recover from transient connection failures with cancellable exponential backoff, and successful batches reset their retry budget. `Sandbox.Terminate()` no longer detaches and instead relies on idle cleanup or an explicit `Detach()` call to release the dedicated connection.
- Added `MountImage`, `UnmountImage`, and `SnapshotDirectory` to experimental Sandbox Sidecar containers. Directory snapshot Images can be used anywhere an Image is accepted, including as mounts and as container filesystems.
- A Sandbox's `Stdout`/`Stderr`, and those of a `ContainerProcess`, no longer read ahead of the caller: output is fetched by the read that asks for it, with no background goroutine and no buffer in between. Wrap them in a `bufio.Reader` for read-ahead.
