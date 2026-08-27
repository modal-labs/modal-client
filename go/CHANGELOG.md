# Go SDK Release Notes

## Latest

### 0.9.0 (2026-07-09)

- [`Functions.FromName`](/docs/sdk/go/latest/Function#fromname) and [`Cls.FromName`](/docs/sdk/go/latest/Cls#fromname) now accept an optional `Version` parameter to look up a version-pinned Function or Cls.
- [`Secrets.FromMap`](/docs/sdk/go/latest/Secret#frommap) is now lazy, so the returned `Secret` has an empty `SecretID` until it is first used. Latency for [`Sandbox.ExperimentalCreate`](/docs/sdk/go/latest/Sandbox#experimentalcreate) and [`Sandbox.Exec`](/docs/sdk/go/latest/Sandbox#exec) is improved by sending the secrets directly to the Sandbox and avoiding the secret creation limits.
- [`Function.WithOptions`](/docs/sdk/go/latest/Function#withoptions) and [`Cls.WithOptions`](/docs/sdk/go/latest/Cls#withoptions) now accept a `RoutingRegion` option to override the region the Function's or Cls's inputs and outputs are routed through.
- Added [`Sandbox.Filesystem.Watch`](/docs/sdk/go/latest/Sandbox#filesystemwatch) to watch a path in the Sandbox for filesystem changes. It yields [`FileWatchEvent`](/docs/sdk/go/latest/FileWatchEvent) objects and supports `Recursive`, an event-type `Filter`, and a `Timeout` bound.
- **Breaking:** [`Sandbox.ReloadVolumes`](/docs/sdk/go/latest/Sandbox#reloadvolumes) now blocks until the Volumes have been reloaded, bounded by a new timeout (55 seconds by default) that is configurable via `SandboxReloadVolumesParams.Timeout`. If the reload does not complete within that window, a [`TimeoutError`](/docs/sdk/go/latest/Errors#timeouterror) is raised.
- **Breaking:** Renamed the `VolumeMountOptions` type to `VolumeMountOptionsParams`. Update [`WithMountOptions(&VolumeMountOptions{...})`](/docs/sdk/go/latest/Volume#withmountoptions) call sites accordingly.
- **Breaking:** Changed `OutboundCIDRAllowlist` and `OutboundDomainAllowlist` in `SandboxCreateParams` from `[]string` to [`*Allowlist`](/docs/sdk/go/latest/Allowlist). A non-nil `*Allowlist` enables allowlist mode (even with empty `Entries`, which blocks all traffic of that type); `nil` means open access. Migrate `OutboundCIDRAllowlist: []string{"10.0.0.0/8"}` to `OutboundCIDRAllowlist: &Allowlist{Entries: []string{"10.0.0.0/8"}}`.

## 0.8

### 0.8.2 (2026-06-29)

- Added [`Sandbox.UpdateNetworkPolicy`](/docs/sdk/go/latest/Sandbox#updatenetworkpolicy) to update the outbound network policy of a running Sandbox. Both `OutboundCIDRAllowlist` and `OutboundDomainAllowlist` must be provided.

### 0.8.1 (2026-06-25)

- [`Sandbox.CreateConnectToken`](/docs/sdk/go/latest/Sandbox#createconnecttoken) now accepts an optional `Port` parameter on its params struct, controlling which container port requests are routed to when using the token. Defaults to 8080.

### 0.8.0 (2026-06-10)

This release primarily contains a number of breaking changes as we continue working towards 1.0.

- Added support for named Images, akin to a Modal-native Image registry, decoupling Image builds from App deployment or Sandbox creation, via [`Image.Publish`](/docs/sdk/go/latest/Image#publish) and [`Image.FromName`](/docs/sdk/go/latest/Image#fromname).
- Added support for restricting the _domains_ that processes inside of a Sandbox can connect to, via `OutboundDomainAllowlist` in [`SandboxCreateParams`](/docs/sdk/go/latest/Sandbox#create).
- Added support for dynamic configuration of Functions: [`Function.WithOptions`](/docs/sdk/go/latest/Function#withoptions), [`Function.WithConcurrency`](/docs/sdk/go/latest/Function#withconcurrency), [`Function.WithBatching`](/docs/sdk/go/latest/Function#withbatching), and [`Function.Instance`](/docs/sdk/go/latest/Function#instance).
- Improved reliability when uploading or downloading large data payloads for Function calls.
- Fixed a bug where adding Dockerfile commands to an Image loaded with [`Image.FromID`](/docs/sdk/go/latest/Image#fromid) could fail to use the resolved Image as the base.
- [`Sandbox.Exec`](/docs/sdk/go/latest/Sandbox#exec) now rejects a relative `Workdir` client-side with [`InvalidError`](/docs/sdk/go/latest/Errors#invaliderror).
- [`Sandbox.Create`](/docs/sdk/go/latest/Sandbox#create) and [`Sandbox.Exec`](/docs/sdk/go/latest/Sandbox#exec) now return an [`InvalidError`](/docs/sdk/go/latest/Errors#invaliderror) if any `Secrets` entry is nil.
- **Breaking:** The Go SDK now reads the [Image Builder Version](https://modal.com/docs/guide/images#image-builder-updates) from your [Modal workspace settings](https://modal.com/settings/image-config), like the Python SDK. Previously, the Go SDK was hardcoded to use version `2024.10`. If your workspace configuration uses a different version, your Images will rebuild once (then be cached as usual), so beware that the first run after upgrading may take longer than usual. Note that version `2025.06` has a number of improvements that are specifically oriented towards Sandbox workflows.
- **Breaking:** [`Sandbox.SnapshotFilesystem`](/docs/sdk/go/latest/Sandbox#snapshotfilesystem) no longer takes a positional `timeout` argument. The timeout now lives on the params struct as `Timeout time.Duration`, bringing the method to parity with [`Sandbox.SnapshotDirectory`](/docs/sdk/go/latest/Sandbox#snapshotdirectory). Migrate `sb.SnapshotFilesystem(ctx, 30*time.Second, params)` to `sb.SnapshotFilesystem(ctx, &SandboxSnapshotFilesystemParams{Timeout: 30*time.Second, ...})`.
- **Breaking:** [`Sandbox.SnapshotFilesystem`](/docs/sdk/go/latest/Sandbox#snapshotfilesystem) and [`Sandbox.SnapshotDirectory`](/docs/sdk/go/latest/Sandbox#snapshotdirectory) now accept an explicit `TTL` field (a `time.Duration`) on their params structs, controlling how long the resulting Image is retained. Both methods default to 30 days. This is a change of default for `SnapshotFilesystem`, which previously kept Images indefinitely. Pass `TTL: modal.NoExpiryTTL` to opt out of expiry.
- **Breaking:** [`Sandbox.SnapshotDirectory`](/docs/sdk/go/latest/Sandbox#snapshotdirectory) now also has a `Timeout` field on its params struct with a default of 55s, which brings it to parity with filesystem snapshots. If the snapshot does not return within that window, a [`TimeoutError`](/docs/sdk/go/latest/Errors#timeouterror) is raised. The timeout can be set arbitrarily high to preserve the old behavior of not timing out.
- **Breaking:** [`Sandbox.Filesystem`](/docs/sdk/go/latest/Sandbox#sandboxfilesystem) is now a field rather than a method. Migrate `sb.Filesystem().ReadText(...)` to `sb.Filesystem.ReadText(...)`.
- **Breaking:** [`Sandbox.FromID`](/docs/sdk/go/latest/Sandbox#fromid) no longer checks if the sandbox ID exists. You can run [`Poll`](/docs/sdk/go/latest/Sandbox#poll) to get the status of your sandbox.
- **Breaking:** Removed the deprecated low-level file-handle API: `Sandbox.Open` and `SandboxFile`. Use the [Sandbox Filesystem API](/docs/sdk/go/latest/Sandbox#sandboxfilesystem) instead: `sandbox.Filesystem`.
- **Breaking:** Removed the deprecated `Volume.ReadOnly` and `Volume.IsReadOnly`. Use [`WithMountOptions(&VolumeMountOptions{ReadOnly: &t})`](/docs/sdk/go/latest/Volume#withmountoptions) instead.
- **Breaking:** All public methods now end with a `*XxxParams` pointer argument, enabling forward-compatibility for future options without additional signature churn. Pass `nil` to accept defaults. Affected methods:
  - [`FunctionCall.FromID`](/docs/sdk/go/latest/FunctionCall#fromid) → `FromID(ctx, id, *FunctionCallFromIDParams)`
  - [`Image.FromID`](/docs/sdk/go/latest/Image#fromid) → `FromID(ctx, id, *ImageFromIDParams)`
  - [`Image.FromRegistry`](/docs/sdk/go/latest/Image#fromregistry) → `FromRegistry(tag, *ImageFromRegistryParams)`, the `*Secret` field remains inside `ImageFromRegistryParams` (signature unchanged)
  - [`Image.FromAwsEcr`](/docs/sdk/go/latest/Image#fromawsecr) → `FromAwsEcr(tag, secret, *ImageFromAwsEcrParams)` — the `*Secret` argument is now a positional parameter preceding the params struct
  - [`Image.FromGcpArtifactRegistry`](/docs/sdk/go/latest/Image#fromgcpartifactregistry) → `FromGcpArtifactRegistry(tag, secret, *ImageFromGcpArtifactRegistryParams)` — the `*Secret` argument is now a positional parameter preceding the params struct
  - [`Image.Build`](/docs/sdk/go/latest/Image#build) → `Build(ctx, app, *ImageBuildParams)`
  - [`Function.GetCurrentStats`](/docs/sdk/go/latest/Function#getcurrentstats) → `GetCurrentStats(ctx, *FunctionGetCurrentStatsParams)`
  - [`ContainerProcess.Wait`](/docs/sdk/go/latest/ContainerProcess#wait) → `Wait(ctx, *ContainerProcessWaitParams)`
  - [`Sandbox.FromID`](/docs/sdk/go/latest/Sandbox#fromid) → `FromID(ctx, id, *SandboxFromIDParams)`
  - [`Sandbox.Wait`](/docs/sdk/go/latest/Sandbox#wait) → `Wait(ctx, *SandboxWaitParams)`
  - [`Sandbox.WaitUntilReady`](/docs/sdk/go/latest/Sandbox#waituntilready) → `WaitUntilReady(ctx, timeout, *SandboxWaitUntilReadyParams)`
  - [`Sandbox.Tunnels`](/docs/sdk/go/latest/Sandbox#tunnels) → `Tunnels(ctx, timeout, *SandboxTunnelsParams)`
  - [`Sandbox.MountImage`](/docs/sdk/go/latest/Sandbox#mountimage) → `MountImage(ctx, path, image, *SandboxMountImageParams)`
  - [`Sandbox.UnmountImage`](/docs/sdk/go/latest/Sandbox#unmountimage) → `UnmountImage(ctx, path, *SandboxUnmountImageParams)`
  - [`Sandbox.SnapshotDirectory`](/docs/sdk/go/latest/Sandbox#snapshotdirectory) → `SnapshotDirectory(ctx, path, *SandboxSnapshotDirectoryParams)`
  - [`Sandbox.SnapshotFilesystem`](/docs/sdk/go/latest/Sandbox#snapshotfilesystem) → `SnapshotFilesystem(ctx, *SandboxSnapshotFilesystemParams)`
  - [`Sandbox.Poll`](/docs/sdk/go/latest/Sandbox#poll) → `Poll(ctx, *SandboxPollParams)`
  - [`Sandbox.SetTags`](/docs/sdk/go/latest/Sandbox#settags) → `SetTags(ctx, tags, *SandboxSetTagsParams)`
  - [`Sandbox.GetTags`](/docs/sdk/go/latest/Sandbox#gettags) → `GetTags(ctx, *SandboxGetTagsParams)`

## 0.7

### 0.7.6 (2026-05-21)

- Added the [Sandbox Filesystem API](/docs/sdk/go/latest/Sandbox#sandboxfilesystem), available via `sandbox.Filesystem()`. The filesystem API contains methods:
  - [`fs.WriteText`](/docs/sdk/go/latest/Sandbox#filesystemwritetext): Write UTF-8 to a file in the Sandbox.
  - [`fs.WriteBytes`](/docs/sdk/go/latest/Sandbox#filesystemwritebytes): Write binary content to a file in the Sandbox.
  - [`fs.ReadText`](/docs/sdk/go/latest/Sandbox#filesystemreadtext): Read a file from the Sandbox and return its contents as a UTF-8 string.
  - [`fs.ReadBytes`](/docs/sdk/go/latest/Sandbox#filesystemreadbytes): Read a file from the Sandbox and return its contents as bytes.
  - [`fs.MakeDirectory`](/docs/sdk/go/latest/Sandbox#filesystemmakedirectory): Create a new directory in the Sandbox.
  - [`fs.ListFiles`](/docs/sdk/go/latest/Sandbox#filesystemlistfiles): List files and directories in a Sandbox directory.
  - [`fs.Stat`](/docs/sdk/go/latest/Sandbox#filesystemstat): Return metadata for a single file, directory, or symlink in the Sandbox.
  - [`fs.CopyFromLocal`](/docs/sdk/go/latest/Sandbox#filesystemcopyfromlocal): Copy a local file into the Sandbox.
  - [`fs.CopyToLocal`](/docs/sdk/go/latest/Sandbox#filesystemcopytolocal): Copy a file from the Sandbox to a local path.
  - [`fs.Remove`](/docs/sdk/go/latest/Sandbox#filesystemremove): Remove a file or directory in the Sandbox.
- Added [`Volume.WithMountOptions`](/docs/sdk/go/latest/Volume#withmountoptions) to configure mount-time options (`ReadOnly` and `SubPath`) when attaching a Volume to a Function or Sandbox. The `SubPath` option mounts a subdirectory of the Volume instead of its root. Calling `WithMountOptions` multiple times on the same Volume stacks: fields left unset preserve their previous value.
- Deprecated `Volume.ReadOnly` in favor of [`WithMountOptions(&VolumeMountOptions{ReadOnly: &t})`](/docs/sdk/go/latest/Volume#withmountoptions). The old method still works but will be removed in a future release.
- Deprecated `Volume.IsReadOnly`; track configured mount options at the call site that configured them instead.

### 0.7.5 (2026-05-19)

- We've improved the reliability of the [`Sandbox.SnapshotFilesystem`](/docs/sdk/go/latest/Sandbox#snapshotfilesystem) operation, especially for large snapshots, and we now support setting a `Timeout` longer than 55s when necessary.
- Added [`Sandbox.UnmountImage`](/docs/sdk/go/latest/Sandbox#unmountimage) to remove an image mount from a path in the Sandbox filesystem and reveal the underlying directory again.
- [`Sandboxes.Create`](/docs/sdk/go/latest/Sandbox#create) now accepts a `Tags` parameter to attach key-value tags to the Sandbox at creation time.
- [`Sandboxes.Create`](/docs/sdk/go/latest/Sandbox#create) now accepts an `InboundCIDRAllowlist` parameter to restrict which source IPs can connect inbound to a sandbox's tunnels and connection tokens.
- Renamed `CIDRAllowlist` to `OutboundCIDRAllowlist` to distinguish from the corresponding inbound allowlist.
- The Go SDK can now respond more gracefully to server throttling (e.g., rate limiting) by backing off and automatically retrying.

### 0.7.4 (2026-04-03)

- [`Sandboxes.Create`](/docs/sdk/go/latest/Sandbox#create) now accepts an `IncludeOidcIdentityToken` parameter. When enabled, a `MODAL_IDENTITY_TOKEN` environment variable is injected into the sandbox, enabling OIDC-based authentication (e.g., for AWS federation).
- We're introducing a concept of "readiness probes" for `Sandbox`. This feature lets you configure a readiness check on a TCP port, with [`NewTCPProbe`](/docs/sdk/go/latest/Probe#newtcpprobe), or by executing a process, with [`NewExecProbe`](/docs/sdk/go/latest/Probe#newexecprobe). Calling [`sb.WaitUntilReady()`](/docs/sdk/go/latest/Sandbox#waituntilready) will block until the Probe succeeds.

### 0.7.3 (2026-03-12)

- Migrated the SDK from `github.com/modal-labs/libmodal` to `github.com/modal-labs/modal-client`.

### 0.7.2 (2026-02-26)

- Updated `Sandbox` methods to wait for newly created sandboxes to be ready and not error immediately when it's not available yet.

### 0.7.1 (2026-02-23)

- No changes affecting the Go SDK.

### 0.7.0 (2026-02-23)

- Added [`Sandbox.MountImage`](/docs/sdk/go/latest/Sandbox#mountimage), which mounts an Image to a path in the Sandbox's filesystem.
- Added [`Sandbox.SnapshotDirectory`](/docs/sdk/go/latest/Sandbox#snapshotdirectory), which snapshots a directory in the running sandbox and creates a new Image from it.
- Upgraded [`Sandbox.Exec`](/docs/sdk/go/latest/Sandbox#exec), bringing improved performance and reliability.
- Added a [`Sandbox.Detach`](/docs/sdk/go/latest/Sandbox#detach) to disconnect your client from the sandbox and clean up any resources associated with the connection. We **strongly recommend** calling `Detach` after you are done interacting with the sandbox. `Detach` does not close streams from `Sandbox.Stdout`. These streams should be closed using their `Close` method.
- [`Sandbox.Terminate`](/docs/sdk/go/latest/Sandbox#terminate) detaches by default. To interact with a running sandbox, use [`Sandboxes.FromID`](/docs/sdk/go/latest/Sandbox#fromid) to create a new Sandbox object.
- [`Sandbox.Terminate`](/docs/sdk/go/latest/Sandbox#terminate) now accepts a `Wait` parameter to wait for the sandbox to terminate and return the exit code.

**Breaking changes:**

- Changed `Sandbox.Terminate`, which now returns `(int, error)`. The `int` is the return code when `&SandboxTerminateParams{Wait: true}` is passed in.
- Added a `Sandbox.Detach` to disconnect your client from the sandbox and clean up any resources associated with the connection. We **strongly recommend** calling `Detach` after you are done interacting with the sandbox.

## 0.6

### 0.6.3 (2026-02-18)

- Fixed a bug where deleting a Volume, Queue, or Secret with `AllowMissing: true` could still raise a `NOT_FOUND` error.
- Improved handling of degraded HTTP/2 connections, which addresses intermittent RST_STREAM errors.

### 0.6.2 (2026-02-09)

- No changes affecting the Go SDK.

### 0.6.1 (2026-01-30)

- Added custom domains to [`Sandboxes.Create`](/docs/sdk/go/latest/Sandbox#create). Note that Sandbox custom domains work differently from Function custom domains and must currently be set up manually by Modal; please get in touch if this feature interests you.

### 0.6.0 (2025-12-10)

- Added `enable_docker` experimental option to `Sandbox.Create`.

## 0.5

### 0.5.6 (2025-12-02)

- Added [`Sandbox.CreateConnectToken`](/docs/sdk/go/latest/Sandbox#createconnecttoken).

### 0.5.5 (2025-11-25)

- Enabled goroutine leak detection for all tests by default.
- Fixed a few remaining goroutine leaks.
- Test clean-ups: ensure we always terminate Sandboxes, close ephemeral objects, etc.
- Added debug logging to [`CloudBucketMount`](/docs/sdk/go/latest/CloudBucketMount) creation, bringing it in line with the JS SDK.
- Aligned the way the Go SDK handles empty/missing fields in gRPC messages, so the behavior is identical to the Python SDK.

### 0.5.4 (2025-11-10)

- Enabled [goleak](https://github.com/uber-go/goleak) for goroutine leak detection in tests.
- Fixed all detected goroutine leaks in Sandboxes and Images.
- Added deletion methods for [`Volume`](/docs/sdk/go/latest/Volume#delete) and [`Secret`](/docs/sdk/go/latest/Secret#delete) objects and updated the deletion methods on [`Queue`](/docs/sdk/go/latest/Queue#delete) objects to support idempotent deletion via the `AllowMissing` parameter.

### 0.5.3 (2025-11-08)

- Fixed a bug where `Sandbox.Exec` would leak goroutines.

### 0.5.2 (2025-11-04)

- Allow adding custom gRPC interceptors when creating a Modal client, to allow instrumentation, custom telemetry, etc.

### 0.5.1 (2025-11-03)

- All Go SDK functions that take a Context will respect the timeout of the context.
- Improved the error message when calling a webhook Function as a normal Function.
- Allow customizing the config file path via `MODAL_CONFIG_PATH` environment variable (defaults to `~/.modal.toml`).
- Added support for passing `MODAL_LOGLEVEL=debug` environment variable to also log debug logs, incl. all gRPC calls, etc.

### 0.5.0 (2025-10-28)

The first beta release of the Modal SDK for Go (graduating from alpha). See the [Migration Guide](../MIGRATION-GUIDE.md) for a detailed list of breaking changes.

- The SDK now exposes a central [`Client`](/docs/sdk/go/latest/Client) object as the main entry point for interacting with Modal resources.
- The interface for working with Modal object instances (Functions, Sandboxes, Images, etc.) is largely the same as before, with some naming changes.
- Calling deployed Functions and classes now uses a new protocol for payload serialization which requires the deployed apps to use the Modal Python SDK 1.2 or newer.
- Internally removed the global client (and config/profile data in global scope), moving all that to the Client type.
- Consistent parameter naming across the Go and JS SDKs: all `Options` structs renamed to `Params`.
- Changed how we do context passing, so contexts now only affect the current operation and are not used for lifecycle management of the created resources.
- All `Params` structs are now passed as pointers for consistency and to support optional parameters.
- Field names follow Go casing conventions (e.g., `Id` → `ID`, `Url` → `URL`, `TokenId` → `TokenID`).
- Added explicit unit suffixes to all parameters that represent memory amounts (in MiB, suffixed with `MiB`).

Additional new features:

- Added support for setting CPU and memory limits when creating Sandboxes and Cls instances.

## 0.0

### 0.0.25 (2025-10-08)

- Internal updates for how authentication tokens are handled for input plane clients.

### 0.0.24 (2025-09-19)

- Added an `Env` parameter to several methods, as a convenience for passing environment variables into Sandboxes, etc.
- Added [`Sandbox.GetTags()`](/docs/sdk/go/latest/Sandbox#gettags).

### 0.0.23 (2025-09-15)

- Added support for PTYs in Sandboxes.

### 0.0.22 (2025-09-11)

- Added [`ImageDockerfileCommands()`](/docs/sdk/go/latest/Image#dockerfilecommands).

### 0.0.21 (2025-09-08)

- Added support for setting idle timeout when creating Sandboxes.

### 0.0.20 (2025-09-02)

- Added `ImageDelete()`.
- Changed `NewImageFromId()` to return a `NotFoundError` if the Image does not exist. Note that the signature of `NewImageFromId()` has changed.

### 0.0.19 (2025-08-26)

- No changes affecting the Go SDK.

### 0.0.18 (2025-08-26)

- Added [`Image.Build`](/docs/sdk/go/latest/Image#build).
- Added `NewImageFromId`.
- Operations on an ephemeral Queue after having called `CloseEphemeral()` will now explicitly fail.
- Added support for instantiating a Cls with custom options, using [`Cls.WithOptions()`](/docs/sdk/go/latest/Cls#withoptions)/[`.WithConcurrency()`](/docs/sdk/go/latest/Cls#withconcurrency)/[`.WithBatching()`](/docs/sdk/go/latest/Cls#withbatching).
- Added support for [Named Sandboxes](https://modal.com/docs/guide/sandbox#named-sandboxes) (example in [`examples/sandbox-named/main.go`](./examples/sandbox-named/main.go)).
- Added support for `VolumeEphemeral()`.

### 0.0.17 (2025-08-22)

- Added support for more parameters to [`Sandbox.Create()`](/docs/sdk/go/latest/Sandbox#create):
  - `BlockNetwork`: Whether to block all network access from the Sandbox.
  - `CIDRAllowlist`: List of CIDRs the Sandbox is allowed to access.
  - `GPU`: GPU reservation for the Sandbox (e.g. "A100", "T4:2", "A100-80GB:4").
  - `Cloud`: Cloud provider to run the Sandbox on.
  - `Regions`: Region(s) to run the Sandbox on.
  - `Verbose`: Enable verbose logging.
  - `Proxy`: Connect a Modal Proxy to a Sandbox.
  - `Workdir`: Set the working directory.
- Added support for mounting [`CloudBucketMount`](/docs/sdk/go/latest/CloudBucketMount)s to Sandboxes.
- Added top-level Image objects that are lazy. The Images are built when creating a Sandbox.
  - `NewImageFromRegistry`
  - `NewImageFromAwsEcr`
  - `NewImageFromGcpArtifactRegistry`
- Added `SecretFromMap()` to create a Secret from key-value pairs (like `from_dict()` in Python).
- Added `Name` field to `App`s, `Sandbox`es, `Secret`s, `Volume`s, and `Queue`s.
- Added support for [`Function.GetCurrentStats()`](/docs/sdk/go/latest/Function#getcurrentstats).
- Added support for [`Function.UpdateAutoscaler()`](/docs/sdk/go/latest/Function#updateautoscaler).
- Added support for [`Function.GetWebURL()`](/docs/sdk/go/latest/Function#getweburl).
- Added support for `Volume.ReadOnly()`.
- Added support for [setting tags](/docs/sdk/go/latest/Sandbox#settags) on Sandboxes, and for [listing Sandboxes](/docs/sdk/go/latest/Sandbox#list) (by tag).

### 0.0.16 (2025-08-07)

- Added support for [getting Sandboxes from an ID](/docs/sdk/go/latest/Sandbox#fromid).

### 0.0.15 (2025-07-23)

- Added support for [snapshotting the filesystem of a Sandbox](/docs/sdk/go/latest/Sandbox#snapshotfilesystem).
- Added support for [polling Sandboxes](/docs/sdk/go/latest/Sandbox#poll) to check if they are still running, or get the exit code.
- Added support to execute commands in Sandboxes with Secrets.
- Added support for creating Sandboxes with Secrets.

### 0.0.14 (2025-07-07)

- Added support for setting up [Tunnels](/docs/sdk/go/latest/Sandbox#tunnels) to expose live TCP ports for Sandboxes.

### 0.0.13 (2025-07-03)

- Fixed calls of Cls with experimental `input_plane_region` option.
- Removed `Function.InputPlaneURL` from being exposed as public API.

### 0.0.12 (2025-07-02)

- Added support for passing a Secret to `ImageFromRegistry()` to pull images from private registries.
- Added support for creating Images from Google Artifact Registry with `ImageFromGcpArtifactRegistry()`.
- Added experimental support for calling remote Functions deployed with the `input_plane_region` option in Python.

### 0.0.11 (2025-06-30)

- Added `InitializeClient()` to initialize the client at runtime with credentials.
- The client library no longer panics at startup if no token ID / Secret is provided. Instead, it will return an error when trying to use the client.

### 0.0.10 (2025-06-28)

- Added `Workdir` and `Timeout` options to `ExecOptions` for Sandbox processes.

### 0.0.9 (2025-06-27)

- Added support for Sandbox filesystem.

### 0.0.8 (2025-06-18)

- Added support for creating Images from AWS ECR with `App.ImageFromAwsEcr()`.
- Added support for accessing Modal Secrets with `modal.SecretFromName()`.

### 0.0.7 (2025-06-09)

- Added support for the [`Queue`](/docs/sdk/go/latest/Queue) object to manage distributed FIFO queues.
  - Queues have a similar interface as Python, with [`Put()`](/docs/sdk/go/latest/Queue#put) and [`Get()`](/docs/sdk/go/latest/Queue#get) being the primary methods.
  - You can put structured objects onto Queues, with limited support for the pickle format.
- Added [`InvalidError`](/docs/sdk/go/latest/Errors#invaliderror), [`QueueEmptyError`](/docs/sdk/go/latest/Errors#queueemptyerror), and [`QueueFullError`](/docs/sdk/go/latest/Errors#queuefullerror) to support Queues.
- Options in the Go SDK now take pointer types, and can be `nil` for default values.

### 0.0.6 (2025-05-30)

- Added support for spawning Functions with [`Function.Spawn()`](/docs/sdk/go/latest/Function#spawn).

### 0.0.5 (2025-05-03)

- Added feature for looking up and calling remote classes via the [`Cls`](/docs/sdk/go/latest/Cls) object.
- Removed the initial `ctx context.Context` argument from `Function.Remote()`.

### 0.0.4 (2025-05-02)

- Support calling remote Functions with arguments greater than 2 MiB in byte payload size.

### 0.0.3 (2025-04-29)

- First public release
- Basic [`Function`](/docs/sdk/go/latest/Function), [`Sandbox`](/docs/sdk/go/latest/Sandbox), [`Image`](/docs/sdk/go/latest/Image), and [`ContainerProcess`](/docs/sdk/go/latest/ContainerProcess) support
