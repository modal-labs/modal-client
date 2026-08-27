# JS SDK Release Notes

## Latest

### 0.9.0 (2026-07-09)

- `client.functions.fromName` and `client.cls.fromName` now accept an optional `version` parameter to look up a version-pinned Function or Cls.
- `client.secrets.fromObject` is now lazy, so the returned `Secret` has an empty `secretId` until it is first used. Latency for `Sandbox.experimentalCreate` and `Sandbox.exec` is improved by sending the secrets directly to the Sandbox and avoiding the secret creation limits.
- `Function_.withOptions` and `Cls.withOptions` now accept a `routingRegion` option to override the region the Function's or Cls's inputs and outputs are routed through.
- Added `Sandbox.filesystem.watch` to watch a path in the Sandbox for filesystem changes. It yields `FileWatchEvent` objects and supports `recursive`, an event-type `filter`, and a `timeoutMs` bound.
- **Breaking:** `Sandbox.reloadVolumes` now blocks until the Volumes have been reloaded, bounded by a new timeout (55 seconds by default) that is configurable via `SandboxReloadVolumesParams.timeoutMs`. If the reload does not complete within that window, a `TimeoutError` is raised.
- **Breaking:** Renamed the `VolumeMountOptions` type to `VolumeMountOptionsParams`. Update any `withMountOptions` type annotations accordingly.

## 0.8

### 0.8.2 (2026-06-29)

- Added `Sandbox.updateNetworkPolicy` to update the outbound network policy of a running Sandbox. Both `outboundCidrAllowlist` and `outboundDomainAllowlist` must be provided.

### 0.8.1 (2026-06-25)

- `Sandbox.createConnectToken` now accepts an optional `port` parameter on its params object, controlling which container port requests are routed to when using the token. Defaults to 8080.

### 0.8.0 (2026-06-10)

This release primarily contains a number of breaking changes as we continue working towards 1.0.

- Added support for named Images, akin to a Modal-native Image registry, decoupling Image builds from App deployment or Sandbox creation, via `Image.publish` and `Image.fromName`.
- Added support for restricting the _domains_ that processes inside of a Sandbox can connect to, via `outboundDomainAllowlist` in `Sandbox.create`.
- Added support for dynamic configuration of Functions: `Function_.withOptions`, `Function_.withConcurrency`, `Function_.withBatching`, and `Function_.instance`.
- Improved reliability when uploading or downloading large data payloads for Function calls.
- Fixed a bug where adding Dockerfile commands to an Image loaded with `Image.fromId` could fail to use the resolved Image as the base.
- `Sandbox.exec` now rejects a relative `workdir` client-side with `InvalidError`. An empty-string `workdir` is also rejected (pass `undefined` to use the image default).
- **Breaking:** The JS SDK now reads the [Image Builder Version](https://modal.com/docs/guide/images#image-builder-updates) from your [Modal workspace settings](https://modal.com/settings/image-config), like the Python SDK. Previously, the JS SDK was hardcoded to use version `2024.10`. If your workspace configuration uses a different version, your Images will rebuild once (then be cached as usual), so beware that the first run after upgrading may take longer than usual. Note that version `2025.06` has a number of improvements that are specifically oriented towards Sandbox workflows. Accordingly, the fixed `ModalClient.imageBuilderVersion` attribute has been removed in favor of `ModalClient.getImageBuilderVersion`.
- **Breaking:** `Sandbox.snapshotFilesystem` no longer takes a positional `timeoutMs` argument. The timeout now lives on the params object as `timeoutMs?: number`, bringing the method to parity with `Sandbox.snapshotDirectory`. Migrate `sb.snapshotFilesystem(30000, params)` to `sb.snapshotFilesystem({ timeoutMs: 30000, ...params })`.
- **Breaking:** `Sandbox.snapshotFilesystem` and `Sandbox.snapshotDirectory` now accept an explicit `ttlMs` field (in milliseconds) on their params objects, controlling how long the resulting Image is retained. Both methods default to 30 days. This is a change of default for `snapshotFilesystem`, which previously kept Images indefinitely. Pass `ttlMs: null` to opt out of expiry.
- **Breaking:** `Sandbox.snapshotDirectory` now also has a `timeoutMs` field on its params object with a default of 55s, which brings it to parity with filesystem snapshots. If the snapshot does not return within that window, a `TimeoutError` is raised. The timeout can be set arbitrarily high to preserve the old behavior of not timing out.
- **Breaking:** `Sandbox.fromId` no longer checks if the sandbox ID exists. You can run `poll` to get the status of your sandbox.
- **Breaking:** Removed the deprecated low-level file-handle API: `sandbox.open` and `SandboxFile`. Use the Sandbox Filesystem API instead: `sandbox.filesystem`.
- **Breaking:** Removed the deprecated `volume.readOnly` and `volume.isReadOnly`. Use `withMountOptions({ readOnly: true })` instead.
- **Breaking:** Removed the deprecated `cidrAllowlist` parameter from `sandboxes.create`. Use `outboundCidrAllowlist` instead.
- **Breaking:** Removed the entire deprecated v0.5.0 backwards-compatibility surface: the global `initializeClient()` / `close()` functions and the `ClientOptions` type; the deprecated static factories `App.lookup`, `Function_.lookup`, `Cls.lookup`, `Queue.lookup` / `Queue.ephemeral` / `Queue.delete`, `Volume.fromName` / `Volume.ephemeral`, `Secret.fromName` / `Secret.fromObject`, `Sandbox.fromId` / `Sandbox.fromName` / `Sandbox.list`, `Image.fromId` / `Image.fromRegistry` / `Image.fromAwsEcr` / `Image.fromGcpArtifactRegistry` / `Image.delete`, `Proxy.fromName`, and `FunctionCall.fromId`; the instance shims `app.createSandbox`, `app.imageFromRegistry` / `imageFromAwsEcr` / `imageFromGcpArtifactRegistry`, the public `CloudBucketMount` constructor, and the deprecated `LookupOptions` / `DeleteOptions` / `EphemeralOptions` type aliases. See [`MIGRATION-GUIDE.md`](../MIGRATION-GUIDE.md).

## 0.7

### 0.7.6 (2026-05-21)

- Added the Sandbox Filesystem API, available via `sandbox.filesystem`. The filesystem API contains methods:
  - `fs.writeText`: Write UTF-8 to a file in the Sandbox.
  - `fs.writeBytes`: Write binary content to a file in the Sandbox.
  - `fs.readText`: Read a file from the Sandbox and return its contents as a UTF-8 string.
  - `fs.readBytes`: Read a file from the Sandbox and return its contents as bytes.
  - `fs.makeDirectory`: Create a new directory in the Sandbox.
  - `fs.listFiles`: List files and directories in a Sandbox directory.
  - `fs.stat`: Return metadata for a single file, directory, or symlink in the Sandbox.
  - `fs.copyFromLocal`: Copy a local file into the Sandbox.
  - `fs.copyToLocal`: Copy a file from the Sandbox to a local path.
  - `fs.remove`: Remove a file or directory in the Sandbox.
- Added `volume.withMountOptions` to configure mount-time options (`readOnly` and `subPath`) when attaching a Volume to a Function or Sandbox. The `subPath` option mounts a subdirectory of the Volume instead of its root. Calling `withMountOptions` multiple times on the same Volume stacks: fields left unset preserve their previous value.
- Deprecated `Volume.readOnly` in favor of `withMountOptions({ readOnly: true })`. The old method still works but will be removed in a future release.
- Deprecated `Volume.isReadOnly`; track configured mount options at the call site that configured them instead.

### 0.7.5 (2026-05-19)

- We've improved the reliability of the `Sandbox.snapshotFilesystem` operation, especially for large snapshots, and we now support setting a `timeoutMs` longer than 55s when necessary.
- Added `sandbox.unmountImage` to remove an image mount from a path in the Sandbox filesystem and reveal the underlying directory again.
- `sandboxes.create` now accepts a `tags` parameter to attach key-value tags to the Sandbox at creation time.
- `sandboxes.create` now accepts an `inboundCidrAllowlist` parameter to restrict which source IPs can connect inbound to a sandbox's tunnels and connection tokens.
- Renamed `cidrAllowlist` to `outboundCidrAllowlist` to distinguish from the corresponding inbound allowlist.
- The JS SDK can now respond more gracefully to server throttling (e.g., rate limiting) by backing off and automatically retrying.

### 0.7.4 (2026-04-03)

- `sandboxes.create` now accepts an `includeOidcIdentityToken` parameter. When enabled, a `MODAL_IDENTITY_TOKEN` environment variable is injected into the sandbox, enabling OIDC-based authentication (e.g., for AWS federation).
- We're introducing a concept of "readiness probes" for `Sandbox`. This feature lets you configure a readiness check on a TCP port, with `Probe.withTcp`, or by executing a process, with `Probe.withExec`. Calling `sb.waitUntilReady()` will block until the Probe succeeds.

### 0.7.3 (2026-03-12)

- Migrated the SDK from `github.com/modal-labs/libmodal` to `github.com/modal-labs/modal-client`.

### 0.7.2 (2026-02-26)

- Updated `Sandbox` methods to wait for newly created sandboxes to be ready and not error immediately when it's not available yet.
- Fixed a bug so that canceling `sandbox.stdout` or `sandbox.stderr` cleans up background resources.
- Updated `Sandbox` to raise a better error when the sandbox was terminated.

### 0.7.1 (2026-02-23)

- Fixed regression in `Sandbox.exec` where it threw a `ChannelCredentials` type error.

### 0.7.0 (2026-02-23)

- Added `Sandbox.mountImage`, which mounts an Image to a path in the Sandbox's filesystem.
- Added `Sandbox.snapshotDirectory`, which snapshots a directory in the running sandbox and creates a new Image from it.
- Upgraded `Sandbox.exec`, bringing improved performance and reliability.
- Added a `sandbox.detach` to disconnect your client from the sandbox and clean up any resources associated with the connection. We **strongly recommend** calling `detach` after you are done interacting with the sandbox. `detach` does not close streams from `Sandbox.stdout`. These streams should be closed using their `close` method.
- `Sandbox.terminate` detaches by default. To interact with a running sandbox, use `sandboxes.fromId` to create a new Sandbox object.
- `Sandbox.terminate` now accepts a `wait` parameter to wait for the sandbox to terminate and return the exit code.

**Breaking changes:**

- Added a `sandbox.detach` to disconnect your client from the sandbox and clean up any resources associated with the connection. We **strongly recommend** calling `detach` after you are done interacting with the sandbox.

## 0.6

### 0.6.3 (2026-02-18)

- Fixed a bug where deleting a Volume, Queue, or Secret with `allowMissing=true` could still raise a `NOT_FOUND` error.
- Improved handling of degraded HTTP/2 connections, which addresses intermittent RST_STREAM errors.

### 0.6.2 (2026-02-09)

- Improved reliability for reading streams from `sandbox.stdout` and `sandbox.stderr`.

### 0.6.1 (2026-01-30)

- Added custom domains to `sandboxes.create`. Note that Sandbox custom domains work differently from Function custom domains and must currently be set up manually by Modal; please get in touch if this feature interests you.

### 0.6.0 (2025-12-10)

- Added `enable_docker` experimental option to `Sandbox.create`.

**Breaking changes:**

- Changed Sandbox parameter defaults to be consistent with the Python SDK:
  - Set default Sandbox timeout to 5 minutes (was previously 10 minutes).
  - Leave the Sandbox entrypoint args empty by default (was previously `["sleep", "48h"]`).

## 0.5

### 0.5.6 (2025-12-02)

- Added `Sandbox.createConnectToken`.

### 0.5.5 (2025-11-25)

- Test clean-ups: ensure we always terminate Sandboxes, close ephemeral objects, etc.
- Updated the API for creating `CloudBucketMount`s, using the same `modal.cloudBucketMounts.create()` pattern as other Modal objects, bringing it in line with the Go SDK.
- Aligned the way the JS SDK handles empty/missing fields in gRPC messages, so the behavior is identical to the Python SDK.

### 0.5.4 (2025-11-10)

- Added deletion methods for `Volume` and `Secret` objects and updated the deletion methods on `Queue` objects to support idempotent deletion via the `allowMissing` parameter.

### 0.5.3 (2025-11-08)

- No changes affecting the JS SDK.

### 0.5.2 (2025-11-04)

- Allow adding custom gRPC interceptors when creating a Modal client, to allow instrumentation, custom telemetry, etc.

### 0.5.1 (2025-11-03)

- Improved the error message when calling a webhook Function as a normal Function.
- Allow customizing the config file path via `MODAL_CONFIG_PATH` environment variable (defaults to `~/.modal.toml`).
- Added support for passing `MODAL_LOGLEVEL=debug` environment variable to also log debug logs, incl. all gRPC calls, etc.

### 0.5.0 (2025-10-28)

The first beta release of the Modal SDK for JS (graduating from alpha). See the [Migration Guide](../MIGRATION-GUIDE.md) for a detailed list of breaking changes.

- The SDK now exposes a central Modal Client object as the main entry point for interacting with Modal resources.
- The interface for working with Modal object instances (Functions, Sandboxes, Images, etc.) is largely the same as before, with some naming changes.
- Calling deployed Functions and classes now uses a new protocol for payload serialization which requires the deployed apps to use the Modal Python SDK 1.2 or newer.
- Internally removed the global client (and config/profile data in global scope), moving all that to the Client type.
- Consistent parameter naming across the JS and Go SDKs: all `Options` interfaces renamed to `Params`.
- Added explicit unit suffixes to all parameters that represent durations (in milliseconds, suffixed with `Ms`) or memory amounts (in MiB, suffixed with `MiB`).

Additional new features:

- Added support for setting CPU and memory limits when creating Sandboxes and Cls instances.

## 0.3

### 0.3.25 (2025-10-08)

- Fixed a bug related to unpickling objects from Python (Function calls, Queues, etc.), where integers between 32768 and 65535 were incorrectly decoded as signed integers.
- Internal updates for how authentication tokens are handled for input plane clients.

### 0.3.24 (2025-09-19)

- Added an `env` parameter to several methods, as a convenience for passing environment variables into Sandboxes, etc.
- Added `Sandbox.getTags()`.

### 0.3.23 (2025-09-15)

- Added support for PTYs in Sandboxes.

### 0.3.22 (2025-09-11)

- Added `Image.dockerfileCommands()`.

### 0.3.21 (2025-09-08)

- Added support for setting idle timeout when creating Sandboxes.

### 0.3.20 (2025-09-02)

- Added `Image.delete()`.
- Changed `Image.fromId()` to throw a `NotFoundError` if the Image does not exist.

### 0.3.19 (2025-08-26)

- `Sandbox.exec` now correctly accepts a list of Secrets.

### 0.3.18 (2025-08-26)

- Added `Image.build`.
- Added `Image.fromId`.
- Added support for instantiating a Cls with custom options, using `Cls.withOptions()`/`.withConcurrency()`/`.withBatching()`.
- Added support for [Named Sandboxes](https://modal.com/docs/guide/sandbox#named-sandboxes) (example in [`examples/sandbox-named.ts`](./examples/sandbox-named.ts)).
- Added support for `Volume.ephemeral()`.

### 0.3.17 (2025-08-22)

- Added support for more parameters to `Sandbox.create()`:
  - `blockNetwork`: Whether to block all network access from the Sandbox.
  - `cidrAllowlist`: List of CIDRs the Sandbox is allowed to access.
  - `gpu`: GPU reservation for the Sandbox (e.g. "A100", "T4:2", "A100-80GB:4").
  - `cloud`: Cloud provider to run the Sandbox on.
  - `regions`: Region(s) to run the Sandbox on.
  - `verbose`: Enable verbose logging.
  - `proxy`: Connect a Modal Proxy to a Sandbox.
  - `workdir`: Set the working directory.
- Added support for mounting `CloudBucketMount`s to Sandboxes.
- Added top-level Image objects that are lazy. The Images are built when creating a Sandbox.
  - `Image.fromRegistry`
  - `Image.fromAwsEcr`
  - `Image.fromGcpArtifactRegistry`
- Added `Secret.fromObject()` to create a Secret from key-value pairs (like `from_dict()` in Python).
- Added `name` field to `App`s, `Sandbox`es, `Secret`s, `Volume`s, and `Queue`s.
- Added support for `Function.getCurrentStats()`.
- Added support for `Function.updateAutoscaler()`.
- Added support for `Function.getWebURL()`.
- Added support for `Volume.readOnly()`.
- Added support for setting tags on Sandboxes, and for listing Sandboxes (by tag).

### 0.3.16 (2025-08-07)

- Added support for getting Sandboxes from an ID.

### 0.3.15 (2025-07-23)

- Added support for snapshotting the filesystem of a Sandbox.
- Added support for polling Sandboxes to check if they are still running, or get the exit code.
- Added support to execute commands in Sandboxes with Secrets.
- Added support for creating Sandboxes with Secrets.

### 0.3.14 (2025-07-07)

- Added support for setting up Tunnels to expose live TCP ports for Sandboxes.

### 0.3.13 (2025-07-03)

- Fixed calls of Cls with experimental `input_plane_region` option.

### 0.3.12 (2025-07-02)

- Added support for passing a Secret to `imageFromRegistry()` to pull images from private registries.
- Added support for creating Images from Google Artifact Registry with `imageFromGcpArtifactRegistry()`.
- Added experimental support for calling remote Functions deployed with the `input_plane_region` option in Python.

### 0.3.11 (2025-06-30)

- Added `initializeClient()` to initialize the client at runtime with credentials.
- The client library no longer fails at startup if no token ID / Secret is provided. Instead, it will throw an error when trying to use the client.

### 0.3.10 (2025-06-28)

- Added `workdir` and `timeout` options to `ExecOptions` for Sandbox processes.

### 0.3.9 (2025-06-27)

- Added support for Sandbox filesystem.

### 0.3.8 (2025-06-24)

- Added support for CommonJS format / `require()`. Previously, the JS SDK only supported ESM `import`.

### 0.3.7 (2025-06-18)

- Added support for creating Images from AWS ECR with `App.imageFromAwsEcr()`.
- Added support for accessing Modal Secrets with `Secret.fromName()`.
- Fixed serialization of some pickled objects (negative ints, dicts).

### 0.3.6 (2025-06-09)

- Added support for the `Queue` object to manage distributed FIFO queues.
  - Queues have a similar interface as Python, with `put()` and `get()` being the primary methods.
  - You can put structured objects onto Queues, with limited support for the pickle format.
- Added `InvalidError`, `QueueEmptyError`, and `QueueFullError` to support Queues.
- Fixed a bug that produced incorrect bytecode for bytes objects.

### 0.3.5 (2025-05-30)

- Added support for spawning Functions with `Function_.spawn()`.

### 0.3.4 (2025-05-06)

- Added feature for looking up and calling remote classes via the `Cls` object.

### 0.3.3 (2025-05-02)

- Support calling remote Functions with arguments greater than 2 MiB in byte payload size.

### 0.3.2 (2025-04-29)

- First public release
- Basic `Function`, `Sandbox`, `Image`, and `ContainerProcess` support
