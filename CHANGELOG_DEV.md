# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- Sandboxes are now created on the [next-generation Sandbox backend](/blog/scaling-to-1-million-concurrent-sandboxes-in-seconds) by default, which was previously opt-in via the `MODAL_SANDBOX_V2=1` environment variable. Compared to the V1 backend, it supports substantially higher Sandbox creation rates and concurrent Sandbox counts, and schedules sandboxes faster. No code changes are required, and Sandboxes that use features the new backend does not support (such as GPUs and network file systems) are automatically created on the V1 backend.
- Added `modal.Cluster` for inspecting clustered execution. Use `Cluster.from_context()` inside a cluster or `Cluster.from_id()` to inspect a cluster by ID. `container_ids()` fetches membership ordered by rank, `container_rank()` looks up a member's rank, and `container_ips()` returns addresses from within the cluster.

- Added `modal.clustered` as a public API. `modal.experimental.clustered` is deprecated.
- Classes decorated with `@app.cls()` can no longer define a custom `__init__` constructor (including one inherited from a base class); this now raises an `InvalidError` instead of a deprecation warning. Use [`modal.parameter()`](/docs/guide/parametrized-functions) to parameterize classes and `@modal.enter()` for initialization logic. Classes deployed by older clients with a custom constructor can still be looked up and called.
- `modal shell` now respects the `MODAL_SANDBOX_V2` setting. V2 shells support Function references and `--add-local` mounts.
- Added a `max_concurrency` parameter to `@app.server()` for limiting the number of concurrent requests handled by each container. This hard limit does not affect autoscaling; configure `target_concurrency` separately to scale based on request load.
- Sandboxes now have experimental support for replacing headers in outbound HTTPS requests with secret values that are never visible to the workload, via the new `modal.experimental.OutboundPolicy` configuration object: pass `_experimental_outbound_policy=` to `Sandbox.create` and call `Sandbox._experimental_update_outbound_policy` on a running Sandbox. This API is experimental and may change in the future.
- Constructing a `modal.NetworkFileSystem` now emits a deprecation warning. Use `modal.Volume` instead.
- Proxy token names can now be set at creation time or updated later with `Workspace.proxy_tokens.update(...)` or `modal workspace proxy-tokens update TOKEN_ID name NAME`, and proxy token listings include the token name and creator.
- Removed the deprecated legacy Sandbox filesystem API: `Sandbox.open()`, `Sandbox.ls()`, `Sandbox.mkdir()`, `Sandbox.rm()`, `Sandbox.watch()`, the `modal.file_io.FileIO` type, and `modal.exception.FilesystemExecutionError`. Use the [`Sandbox.filesystem`](/docs/sdk/py/latest/Sandbox#filesystem) APIs instead.
- [`FunctionCall.get_call_graph()`](/docs/sdk/py/latest/FunctionCall#get_call_graph) now returns up to 5000
  nodes instead of 100.
- Volume file downloads no longer transfer the trailing zero bytes of sparse file blocks, which the client now fills in locally.
- Volume file downloads now stop reading a response that runs past the range of the file it is meant to carry, and abandon the whole download as soon as any part of it fails, so that nothing is written to the destination after the error is raised. A download that stops receiving data for 60 seconds is also abandoned and retried; set `MODAL_VOLUME_BLOCK_READ_TIMEOUT` (or the `volume_block_read_timeout` profile key) to change that limit, or to `0` to wait indefinitely.
- Sandbox Sidecars can now snapshot their filesystems into reusable Images with `sidecar.snapshot_filesystem()`.
- Added `mount_image()`, `unmount_image()`, and `snapshot_directory()` to experimental Sandbox Sidecar containers. Directory snapshot Images can be used anywhere an Image is accepted, including as mounts and as container filesystems.
- A client's connection to a Sandbox is now dropped automatically after a period of inactivity, and reconnected if the Sandbox is used again. This makes `Sandbox.detach()` optional. `Sandbox.detach()` does not interrupt or wait for running concurrent operations on the Sandbox; connections are promptly closed once those operations complete. Use `Sandbox.from_id()` to obtain a new handle if you need to continue interacting with the Sandbox.
- Decorator factories such as `@app.function()`, `@app.cls()`, `@modal.method()` and `@modal.enter()` no longer accept a positional argument. Applying them without parentheses (e.g. `@app.function`) now raises a `TypeError` instead of a `modal.exception.InvalidError`, and the decorators are fully typed so strict type checkers (pyright, basedpyright) no longer report them as partially unknown.
- Added the [`Sandbox.logs`](/docs/sdk/py/latest/Sandbox#logs) namespace to retrieve Sandbox entrypoint logs directly from the SDK. The namespace has two different methods, allowing you to `fetch()` logs from a specific date/time range, or `tail()` the most recent logs.
- The `modal.experimental` namespace is now available after `import modal`, without a separate `import modal.experimental`.
- Fixed gRPC channels attempting to reuse connections whose underlying transport is closing.
- Sandbox and `ContainerProcess` output streams now reset their transient-error retry budget when output is received, so the budget limits consecutive failures rather than failures over the stream's lifetime.
- Added `modal app info` CLI command that displays the constituent functions and servers of an app as well as its deployment lifecycle information.
- Added `App.info` method that displays the constituent functions and servers of an app as well as its deployment lifecycle information.
- Added `Function.info` method that displays static information about a particular Function
- Added `Server.info` method that displays static information about a particular Server
- Added `modal function stats` to inspect performance metrics for a Function over a selected time window.
- Added `Function.stats()` and `Server.stats()` methods to inspect performance metrics for a Function or Server over a time window.
- Renamed the return type of `Function.get_current_stats()` to `FunctionCurrentStats`.
- Added an `--expires-in` option to `modal token new` for setting the lifetime of the new token (e.g. `12h`, `7d`).
- Added `modal function logs` to fetch or stream logs from a modal function.
- Added `modal server logs` to fetch or stream logs from a modal server.
- Added `modal server stats` to inspect performance metrics for a Server over a selected time window.
- Removed the following deprecated public API:
  - `Object.deps`, `Object.is_hydrated`, `Object.local_uuid`
  - `Function.from_local`, `Function.get_build_def`, `Function.get_raw_f`, `Function.is_generator`, `Function.spec`, `Function.stub`, `Function.tag`
  - `App.image`, `App.is_interactive`, `App.set_description`
  - `Cls.validate_construction_mechanism`, `Cls.from_local`
- Added SDK support for Sessioned Servers, a primitive built on top of Modal Servers for applications that need multiple HTTP requests to reach the same container. Includes `@modal.sessioned()` to mark a server as sessioned, as well as `Server.sessions.start()` and `Server.sessions.terminate()` to begin and end sessions for a given server.
- We're removing the experimental `modal bootstrap` CLI; use `modal endpoint` to quickly stand up production-ready endpoints.
- Added `modal function variants` to list the variants of a Function. It shows the 200 variants running the most containers by default; pass `-n`/`--limit` to show a different number, or `--limit 0` to list every variant, newest first.
- Added `modal.Function.from_id` and `modal.Server.from_id` methods to look up Functions and Servers from their ID.
- Added `modal function calls` and `modal server requests` to inspect recent invocations.
- Added an `environment_name` field to `modal.SecretInfo`.
- Added `modal.Environment.apps` namespace with a `.list` method for listing all live apps in an environment.
- `Sandbox.create` allows specifying a virtual machine runtime via `runtime="vm"`, which supersedes `experimental_options={"vm_runtime": True}`. Leaving `runtime` unset lets Modal pick the runtime, whilst setting `runtime="gvisor"` explicitly opts-into the gVisor runtime.
- Added a `--runtime vm|gvisor` option to `modal shell` for choosing the runtime of the shell's Sandbox.
- Added a `FunctionInfo.web_info` field, containing info about web functions.
- Added the `FunctionInfo.method_names` and `FunctionInfo.method_details` fields, which for `modal.Cls` instance functions, gives information about what methods are defined on the `Cls` as well as if they are web methods.
- Added `FunctionInfo.image_info` and `ServerInfo.image_info`, which hold metadata about the image used by the calling `Function`/`Server`.
- Added a `FunctionInfo.cluster_info` field, which returns info about functions decorated with `@modal.clustered`.
- Added a `FunctionInfo.batching_info` field, which returns info about functions decorated with `@modal.batched` or created with `.with_batching()`.
- Added a `FunctionInfo.concurrency_info` field, which returns info about functions decorated with `@modal.concurrent` or created with `.with_concurrency()`.
- Added a `ServerInfo.sessioned` field which describes whether or not the Server is sessioned.

## JS

- Sandboxes are now created on the [next-generation Sandbox backend](/blog/scaling-to-1-million-concurrent-sandboxes-in-seconds) by default, which was previously opt-in via the `MODAL_SANDBOX_V2=1` environment variable. Compared to the V1 backend, it supports substantially higher Sandbox creation rates and concurrent Sandbox counts, and schedules sandboxes faster. No code changes are required, and Sandboxes that use features the new backend does not support (such as GPUs) are automatically created on the V1 backend.
- Added a `logs` property to `Sandbox` objects for fetching entrypoint logs over a time range or tailing the most recent entries.
- `SandboxCreateParams` allows specifying a virtual machine runtime via `runtime: "vm"`, which supersedes `experimentalOptions: { vm_runtime: true }`. Leaving `runtime` unset lets Modal pick the runtime, whilst setting `runtime: "gvisor"` explicitly opts-into the gVisor runtime.

## Go
- Sandboxes are now created on the [next-generation Sandbox backend](/blog/scaling-to-1-million-concurrent-sandboxes-in-seconds) by default, which was previously opt-in via the `MODAL_SANDBOX_V2=1` environment variable. Compared to the V1 backend, it supports substantially higher Sandbox creation rates and concurrent Sandbox counts, and schedules sandboxes faster. No code changes are required, and Sandboxes that use features the new backend does not support (such as GPUs) are automatically created on the V1 backend.
- Added a `logs` attribute for `Sandbox` objects for fetching entrypoint logs over a time range or tailing the most recent entries.
- `SandboxCreateParams` allows specifying a virtual machine runtime via `Runtime: modal.SandboxRuntimeVM`, which supersedes `ExperimentalOptions: map[string]any{"vm_runtime": true}`. Leaving `Runtime` unset lets Modal pick the runtime, whilst setting `Runtime: modal.SandboxRuntimeGVisor` explicitly opts-into the gVisor runtime.
