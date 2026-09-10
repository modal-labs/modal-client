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
- Fixed gRPC channels attempting to reuse connections whose underlying transport is closing.
- Sandbox and `ContainerProcess` output streams now reset their transient-error retry budget when output is received, so the budget limits consecutive failures rather than failures over the stream's lifetime.
- Added `modal app info` CLI command that displays the constituent functions and servers of an app as well as its deployment lifecycle information.
- Added `App.info` method that displays the constituent functions and servers of an app as well as its deployment lifecycle information.
- Added `Function.info` method that displays static information about a particular Function
- Added `Server.info` method that displays static information about a particular Server
- Added `modal function stats` to inspect performance metrics for a deployed Function over a selected time window.
- Added `modal function logs` to fetch or stream logs from a modal function.
- Added `modal server logs` to fetch or stream logs from a modal server.

## JS


## Go
