# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

Draft releases notes should be added as part of the PR introducing the change. During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

## JS

- Sandboxes are now created on the [next-generation Sandbox backend](/blog/scaling-to-1-million-concurrent-sandboxes-in-seconds) by default, which was previously opt-in via the `MODAL_SANDBOX_V2=1` environment variable. Compared to the V1 backend, it supports substantially higher Sandbox creation rates and concurrent Sandbox counts, and schedules sandboxes faster. No code changes are required, and Sandboxes that use features the new backend does not support (such as GPUs) are automatically created on the V1 backend.
- Added a `logs` property to `Sandbox` objects for fetching entrypoint logs over a time range or tailing the most recent entries.
- `client.sandboxes.create()` now waits for Sandboxes requesting GPUs to be scheduled before returning. If no capacity becomes available within that time, the Sandbox is terminated and `ResourceExhaustedError` is thrown.
- `SandboxCreateParams` allows specifying a virtual machine runtime via `runtime: "vm"`, which supersedes `experimentalOptions: { vm_runtime: true }`. Leaving `runtime` unset lets Modal pick the runtime, whilst setting `runtime: "gvisor"` explicitly opts-into the gVisor runtime.

Finally, the release includes the following breaking change:

- Renamed the return type of `Function_.getCurrentStats()` from `FunctionStats` to `FunctionCurrentStats`, matching Python.


## Go

- Sandboxes are now created on the [next-generation Sandbox backend](/blog/scaling-to-1-million-concurrent-sandboxes-in-seconds) by default, which was previously opt-in via the `MODAL_SANDBOX_V2=1` environment variable. Compared to the V1 backend, it supports substantially higher Sandbox creation rates and concurrent Sandbox counts, and schedules sandboxes faster. No code changes are required, and Sandboxes that use features the new backend does not support (such as GPUs) are automatically created on the V1 backend.
- Added a `logs` attribute for `Sandbox` objects for fetching entrypoint logs over a time range or tailing the most recent entries.
- `Sandboxes.Create` now waits for Sandboxes requesting GPUs to be scheduled before returning. If no capacity becomes available within that time, the Sandbox is terminated and `modal.ResourceExhaustedError` is returned.
- `SandboxCreateParams` allows specifying a virtual machine runtime via `Runtime: modal.SandboxRuntimeVM`, which supersedes `ExperimentalOptions: map[string]any{"vm_runtime": true}`. Leaving `Runtime` unset lets Modal pick the runtime, whilst setting `Runtime: modal.SandboxRuntimeGVisor` explicitly opts-into the gVisor runtime.

Finally, the release includes the following breaking change:

- Renamed the return type of `Function.GetCurrentStats()` from `FunctionStats` to `FunctionCurrentStats`, matching Python.
