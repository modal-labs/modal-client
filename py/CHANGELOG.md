# Changelog

This changelog documents user-facing updates (features, enhancements, fixes, and deprecations) to the `modal` client library.

## Latest

### 1.4.1 (2026-03-30)

- We're introducing a concept of "readiness probes" for `modal.Sandbox`. This feature lets you configure a readiness check on a TCP port (`modal.Probe.with_tcp()`) or by executing a process (`modal.Probe.with_exec()`). Calling `sb.wait_until_ready()` will block until the Probe succeeds:
  ```python notest
  app = modal.App.lookup('sandbox-app', create_if_missing=True)
  probe = modal.Probe.with_tcp(8080)
  sb = modal.Sandbox.create(
      "python3", "-m", "http.server", "8080",
      readiness_probe=probe,
      app=app,
  )
  sb.wait_until_ready()
  ```
- We fixed a longstanding bug that could cause WebSocket performance to degrade after handling hundreds of connections from the same container.
- We improved the performance of `modal container logs` when fetching logs for an old container.
- We fixed a bug introduced in 1.4.0 that made the `modal` CLI crash on `typer<0.19.0`.

### 1.4.0 (2026-03-25)

We've made significant CLI enhancements so that Modal logs can be more accessible to coding agents:

- The `modal app logs` and `modal container logs` commands now have the ability to fetch historical logs using counting (e.g. `--tail 1000`) or time-based (e.g., `--since 4h`, `--until 2026-03-15`, etc.) configuration. Note that historical log access is subject to plan-level retention limits.
- The `modal container logs` command also accepts `--all` to fetch the complete set of logs for that Container or Sandbox.
- Both CLI commands now accept a `--search` filter, and they can also filter by `--source` (`stdout`/`stderr`/`system`).
- The `modal app logs` command additionally accepts `--function`, `--function-call`, and `--container` filters.
- The `modal app logs` command can prefix each line with the ID of the Function, FunctionCall, or Container where it originated (e.g. `--show-function-id`).
- Note that the default behavior of these commands has changed. Previously, they would follow (i.e., stream) logs by default, but you now must pass `--follow` to get this behavior. The new default will always show the most recent 100 log entries.

We're releasing a new [Sandbox filesystem API](https://modal.com/docs/guide/sandbox-files) (currently in Beta) with significantly improved reliability and ergonomics:

- Use `sb.filesystem.copy_from_local` / `sb.filesystem.copy_to_local` to transfer file contents between your local filesystem and the Sandbox filesystem.
- Use `sb.filesystem.write_text` / `sb.filesystem.read_text` or `sb.filesystem.write_bytes` / `sb.filesystem.read_bytes` to transfer file contents between local memory and the Sandbox filesystem.
- These new APIs replace the `modal.Sandbox.open` method and the `modal.file_io.FileIO` type that it returns; the old APIs are now deprecated.

We're introducing the concept of "deployment strategies" to give you more flexibility over what happens when redeploying your App:

- By passing `modal deploy --strategy recreate` (or `app.deploy(strategy="recreate")` in the SDK), you can immediately terminate any containers that are running when the deployment completes. This is most useful for development workflows, as it guarantees that any subsequent input will be handled by containers running the new version of the App. This trades off some downtime for certainty about when the new version will be in use.
- The disruptive "recreate" strategy is also useful when your App runs at its `max_containers` limit, as otherwise we are unable to bring up replacement capacity.
- The `modal serve` command now uses a "recreate" strategy during code updates.
- The default "rolling" strategy is unchanged. This strategy prioritizes uptime, but means that old containers may still continue handling inputs for some time.

We've also included a number of smaller new features and improvements:

- Sandboxes now accept an `include_oidc_identity_token` parameter in `modal.Sandbox.create`. When set to `True`, a `MODAL_IDENTITY_TOKEN` environment variable will be injected into the Sandbox, enabling OIDC-based authentication (e.g., for AWS federation). See the [OIDC integration guide](https://modal.com/docs/guide/oidc-integration) for more details.
- The new `modal.Image.from_scratch()` constructor creates an empty Image, equivalent to `FROM scratch` in Docker. This is primarily useful as a lightweight filesystem to mount into a Sandbox via `modal.Sandbox.mount_image`.
- The `modal container list` command now accepts an `--app-id` filter to return containers for a specific App.
- We've addressed an issue where `modal.Sandbox.exec` could hang if the Sandbox had terminated immediately after creation.
- An exception is now raised if the *same* Volume or CloudBucketMount is mounted at multiple paths in a container.
- The client will now error faster (≈60s) if it cannot establish an initial connection the Modal servers.

Finally, we are introducing a small number of breaking changes and enforcing some deprecations of pre-1.0 APIs:

- Exceptions returned by `modal.Function.map()` are no longer wrapped in a `UserCodeException` type, and we're deprecating the transitional `wrap_returned_exceptions=` parameter.
- The `modal.enable_output()` context manager no longer yields a value; this had briefly leaked an internal type.
- We've removed unused `namespace` parameters from a number of APIs.
- It's now required to pass `-m` on the CLI when using a module path spelling of the Function reference (e.g. `modal deploy -m project.app`)
- We've removed backwards compatibility for the old autoscaler configuration (`keep_warm`, `concurrency_limit`, etc.).
- It's no longer possible to look up a specific method on a Cls using `modal.Function.from_name`; use `modal.Cls.from_name` instead.

## 1.3

### 1.3.5 (2026-03-03)

- We've added a `modal changelog` CLI for retrieving changelog entries with a flexible query interface (e.g. `modal changelog --since=1.2`, `modal changelog --since=2025-12-01`, `modal changelog --newer`). We expect that this will be a useful way to surface information about new features to coding agents.
- We've added a new `modal.Secret.update` method, which allows you to programmatically modify the environment variables within a Secret. This method has the semantics of Python's `dict.update`: Secret contents can be overwritten or extended when using it. Note that Secret updates will take effect only for containers that start up after the modification.
- The dataclass returned by `modal.Function.get_current_stats()` now includes a `num_running_inputs` field that reports the number of inputs the Function is currently handling.


### 1.3.4 (2026-02-23)

- We're introducing "Directory Snapshots": a new beta feature for persisting specific directories past the lifetime of an individual Sandbox. Using the new methods `modal.Sandbox.snapshot_directory()` and `modal.Sandbox.mount_image()`, you can capture the state of a directory and then later include it in a different Sandbox:
  ```python notest
  sb = modal.Sandbox.create(app=app)
  snapshot = sb.snapshot_directory("/project")

  sb2 = modal.Sandbox.create(app=app)
  sb2.mount_image("/project", snapshot)
  ```
  This feature can be useful for separating the lifecycle of application code in the Sandbox's main Image from project code that changes in each Sandbox session. Files in the mounted snapshot also benefit from several optimizations that allow them to be read faster. See the [Sandbox Snapshot guide](https://modal.com/docs/guide/sandbox-snapshots) for more information.
- We've added a new `modal.Sandbox.detach()` method that we recommend calling after you are done interacting with a Sandbox. This method disconnects your local client from the Sandbox and cleans up resources associated with the connection. After calling `detach`, operations on the Sandbox object may raise and are otherwise not guaranteed to work.
- The `modal.Sandbox.terminate()` method now accepts a `wait` parameter. With `wait=True`, `terminate` will block until the Sandbox is finished and return the exit code. The default `wait=False` maintains the previous behavior.
- Throughput for writing to the `stdin` of a `modal.Sandbox.exec` process has been increased by 8x.
- We've added a new `modal.Volume.from_id()` method for referencing a Volume by its object id.

### 1.3.3 (2026-02-12)

- We've added a new `modal billing report` CLI and promoted the `modal.billing.workspace_billing_report` API to General Availability for all Team and Enterprise plan workspaces.
- We've added `modal.Queue.from_id()` and `modal.Dict.from_id()` methods to support referencing a Queue or Dict by its object id.
- Modal's async usage warnings are now enabled by default. These warnings will fire when using a [blocking interface on a Modal object](https://modal.com/docs/guide/async) in an async context. We've aimed to provide detailed and actionable suggestions for how to modify the code, which makes the warnings verbose. While we recommend addressing any warnings that pop up, as they can point to significant performance issues or bugs, we also provide a configuration option to disable them (`MODAL_ASYNC_WARNINGS=0` or `async_warnings = false` in the `.modal.toml`). Please report any apparent false positives or incorrect suggested fixes.
- We've fixed a bug where the ASGI scope's `state` contents could leak between requests when using `@modal.asgi_app`.

### 1.3.2 (2026-01-30)

- Modal objects now have a `.get_dashboard_url()` method. This method will return a URL for viewing that object on the Modal dashboard:
  ```python
  fc = f.spawn()
  print(fc.get_dashboard_url())  # Easy access to logs, etc.
  ```
- There is also a new `modal dashboard` CLI and new `modal app dashboard` / `modal volume dashboard` CLI subcommands:
  ```bash
  modal dashboard  # Opens up the Apps homepage for the current environment
  modal dashboard <object-id>  # Opens up a view of this object
  modal app dashboard <app-name>  # Opens up the dashboard for this deployed App
  modal volume dashboard <volume-name>  # Opens up the file browser for this persistent Volume
  ```
- You can now pass a Sandbox ID (`sb-xxxxx`) directly to the `modal container logs` CLI.
- The `modal token info` CLI will now include the token name, if provided at token creation.
- We've fixed an issue where `modal.Cls.with_options()` (or the `with_concurrency()` / `with_batching()` methods) could sometimes use stale argument values when called repeatedly.


### 1.3.1 (2026-01-22)

- We've improved our experimental support for Python 3.14t (free-threaded Python) inside Modal containers.
  - The container environment will now use the Python implementation of the Protobuf runtime rather than the incompatible `upb` implementation.
  - As 3.14t images are not being published to the official source for our prebuilt `modal.Image.debian_slim()` images, we recommend using `modal.Image.from_registry` to build a 3.14t Image:
    ```python
    modal.Image.from_registry("debian:bookworm-slim", add_python="3.14t")
    ```
  - Note that 3.14t support is available only on the 2025.06 [Image Builder Version](https://modal.com/settings/image-config).
  - Support is still experimental, so please share any issues that you encounter running 3.14t in Modal containers.
- It's now possible to provide a `custom_domain` for a `modal.Sandbox`:
  ```python
  sb = modal.Sandbox.create(..., custom_domain="sandboxes.mydomain.com")
  ```
  Note that Sandbox custom domains work differently from Function custom domains and must currently be set up manually by Modal; please get in touch if this feature interests you.
- We added a new `modal token info` CLI command to retrieve information about the credentials that are currently in use.
- We added a `--timestamps` flag to a number of CLI entrypoints (`modal run`, `modal serve`, `modal deploy`, and `modal container logs`) to show timestamps in the logging output.
- The automatic CLI creation for `modal run` entrypoints now supports `Literal` type annotations, provided that the literal type contains either all `str` or all `int` values.
- We've fixed a bug that could cause App builds to fail with an uninformative `CancelledError` when the App was misconfigured.
- We've improved client resource management when running `modal.Sandbox.exec`, which avoids a rare thread race condition.

### 1.3.0 (2025-12-19)

Modal now supports Python 3.14. Python 3.14t (the free-threading build) support is currently a work in progress, because we are waiting for dependencies to be updated with free-threaded support. Additionally, Modal no longer supports Python 3.9, which has reached [end-of-life](https://devguide.python.org/versions).

We are adding experimental support for detecting cases where Modal's blocking APIs are used in async contexts (which can be a source of bugs or performance issues). You can opt into runtime warnings by setting `MODAL_ASYNC_WARNINGS=1` as an environment variable or `async_warnings = true` as a config field. We will enable these warnings by default in the future; please report any apparent false positives or other issues while support is experimental.

This release also includes a small number of deprecations and behavioral changes:

- The Modal SDK will no longer propagate `grpclib.GRPCError` types out to the user; our own `modal.Error` subtypes will be used instead. To avoid disrupting user code that has relied on `GRPCError` exceptions for control flow, we are temporarily making some exception types inherit from `GRPCError` so that they will also be caught by `except grpclib.GRPCError` statements. Accessing the `.status` attribute of the exception will issue a deprecation warning, but warnings cannot be issued if the exception object is only caught and there is no other interaction with it. We advise proactively migrating any exception handling to use Modal types, as we will remove the dependency on `grpclib` types entirely in the future. See the [`modal.exception`](https://modal.com/docs/reference/modal.exception) docs for the mapping from gRPC status codes to Modal exception types.
- The `max_inputs` parameter in the `@app.function()` and `@app.cls` decorators has been renamed to `single_use_containers` and now takes a boolean value rather than an integer. Note that only `max_inputs=1` has been supported, so this has no functional implications. This change is being made to reduce confusion with `@modal.concurrent(max_inputs=...)` and so that Modal's autoscaler can provide better performance for Functions with single-use containers.
- The async (`.aio`) interface has been deprecated from `modal.FunctionCall.from_id`, `modal.Image.from_id`, and `modal.SandboxSnapshot.from_id`, because these methods do not perform I/O.
- The `replace_bytes` and `delete_bytes` methods have been removed from the `modal.file_io` filesystem interface.
- Images built with `modal.Image.micromamba()` using the 2023.12 [Image Builder Version](https://modal.com/docs/guide/images#image-builder-updates) will now use a Python version that matches their local environment by default, rather than defaulting to Python 3.9.

## 1.2

### 1.2.6 (2025-12-16)

- Fixed bug where iterating on a `modal.Sandbox.exec` output stream could raise unauthenticated errors.

### 1.2.5 (2025-12-12)

- It is now possible to set a custom `name=` for a Function without using `serialized=True`. This can be useful when decorating a function multiple times, e.g. applying multiple Modal configurations to the same implementation.
- It is now possible to start `modal shell` with a Modal Image ID (`modal shell im-abc123`). Additionally, `modal shell` will now warn if you pass invalid combinations of arguments (like `--cpu` together with the ID of an already running Sandbox, etc.).
- Fixed a bug in `modal shell` that caused e.g. `vi` to fail with unicode decode errors.
- Fixed a thread-safety issue in `modal.Sandbox` resource cleanup.
- Improved performance when adding large local directories to an Image.
- Improved async Sandbox performance by not blocking the event loop while reading from `stdout` or `stderr`.

### 1.2.4 (2025-11-21)

- Fixed a bug in `modal.Sandbox.exec` when using `stderr=StreamType.STDOUT` (introduced in v1.2.3).
- Added a new `h2_enabled` option in `modal.forward`, which enables HTTP/2 advertisement in TLS establishment.

### 1.2.3 (2025-11-20)

- CPU Functions can now be configured to run on non-preemptible capacity by setting `nonpreemptible=True` in the `@app.function()` or `@app.cls()` decorator. This feature is not currently available when requesting a GPU. Note that non-preemptibility incurs a 3x multiplier on CPU and memory pricing. See the [Guide](https://modal.com/docs/guide/preemption) for more information on preemptions.
- The Modal client can now respond more gracefully to server throttling (e.g., rate limiting) by backing off and automatically retrying. This behavior can be controlled with a new `MODAL_MAX_THROTTLE_WAIT` config variable. Setting the config to `0` will preserve the previous behavior and treat rate limits as an exception; setting it to a nonzero number (the unit is seconds) will allow a limited duration of retries.
- The `modal.Sandbox.exec` implementation has been rewritten to be more reliable and efficient.
- Added a new `--add-local` flag to `modal shell`, allowing local files and directories to be included in the shell's container.
- Fixed a bug introduced in v1.2.2 where some Modal objects (e.g., `modal.FunctionCall`) were not usable after being captured in a Memory Snapshot. The bug would result in a `has no loader function` error when the object was used.

### 1.2.2 (2025-11-10)

- `modal.Image.run_commands` now supports `modal.Volume` mounts. This can be helpful for accelerating builds by keeping a package manager cache on the Volume:

  ```python
  cache_vol = modal.Volume.from_name("cache-mount")
  cmd_using_cache = "..."
  image = modal.Image.debian_slim().run_commands(cmd_using_cache, volumes={"/cache": cache_vol})
  ```

- All Modal objects now accept an optional `modal.Client` object in their constructor methods. Passing an explicit client can be helpful in cases where Modal credentials are retrieved from within the Python process that is making requests.
- The `name=` passed to `modal.Sandbox.create` and `modal.Sandbox.from_name` is now required to follow other Modal object naming rules (must contain only alphanumeric characters, dashes, periods, or underscores and cannot exceed 64 characters). Passing an invalid name will now error.
- `modal.CloudBucketMount` now supports `force_path_style=True` to disable virtual-host-style addressing. See [mountpoint-s3 endpoints docs](https://github.com/awslabs/mountpoint-s3/blob/main/doc/CONFIGURATION.md#endpoints-and-aws-privatelink) for details.
- The output from `modal config show` is now valid JSON and can be parsed by CLI tools such as `jq`.
- Fixed a bug where App tags were not attached to Image builds that occur when first deploying the App.

### 1.2.1 (2025-10-22)

- It's now possible to override the default `-dev` suffix applied to the autogenerated URLs for ephemeral Apps (i.e., when using `modal serve`) via a new `dev_suffix` field in the `.modal.toml` config file, or equivalently with the `MODAL_DEV_SUFFIX` environment variable. This can help avoid collisions when multiple users of a workspace are working on the same codebase simultaneously.
- Fixed a bug where reading long stdout/stderr from `modal.Sandbox.exec()` could break in `text=True` mode.
- Fixed a bug where the status code was not checked when downloading a file from a Volume.
- `modal run --detach ...` will now exit more gracefully if you lose internet connection while your App is running.

### 1.2.0 (2025-10-09)

In this release, we're introducing the concept of "App tags", which are simple key-value metadata that can be included to provide additional organizational context. Tags can be defined as part of the `modal.App` constructor:

```python
app = modal.App("llm-inference-server", tags={"team": "genai-platform"})
```

Tags can also be added to an active App via the new `modal.App.set_tags()` method, and current tags can be retrieved with the new `modal.App.get_tags()`.

This release also introduces a new API for generating a tabular billing report: `modal.billing.workspace_billing_report()`. The billing API will report the cost incurred by each App, aggregated over time intervals (currently supporting a daily or hourly resolution). The report can optionally include App tags, allowing you to perform cost allocation using your own organizational schema.

Note that the initial release of the billing API is a private beta. Please get in touch to discuss access.

This release also includes some internal changes to Function input/output serialization. These changes will provide better support for calling into Modal Functions from our `modal-js` and `modal-go` SDKs. Versions 0.4 or later of `modal-js` and `modal-go` will only be able to invoke Functions in Apps deployed with version 1.2 or later of the Python SDK.

Other new features and improvements:

- The new `modal.Sandbox.create_connect_token()` method facilitates authentication for making HTTP / Websocket requests to a server running in a Sandbox:

  ```python notest
  sb = modal.Sandbox.create(...)

  # Create a connect token, optionally including arbitrary user metadata
  creds = sb.create_connect_token(user_metadata={"user_id": "user123"})

  # Make an http request, passing the token in the authorization header
  requests.get(creds.url, headers={"Authorization": f"Bearer {creds.token}"})
  ```

  See the [Sandbox Networking guide](https://modal.com/docs/guide/sandbox-networking) for more information.

- The new `modal.Image.build()` method allows you to eagerly trigger an Image build. This is particularly helpful when working with Sandboxes, as otherwise the Image build would happen lazily inside `modal.Sandbox.create()`:

  ```python notest
  app = modal.App.lookup("sandbox-app")
  image = modal.Image.from_registry("ubuntu")

  # This step will block until the build completes
  image.build(app)

  # Now the Sandbox will be created and scheduled immediately
  sb = modal.Sandbox.create(app=app, image=image)
  ```

- We've added an `env` parameter to a number of methods that configure Function, Sandbox, or Image execution. This parameter accepts a dictionary and adds the contents as environment variables in the relevant Modal container. This allows for simpler inclusion of non-sensitive information compared to using a `modal.Secret`.
- It's now possible to pass a `modal.CloudBucketMount` instance to the `volumes=` parameter of `modal.Cls.with_options` (previously, only dynamic addition of `modal.Volume` mounts was supported).
- The new `modal.Sandbox.get_tags()` method will fetch the tags currently in use by the Sandbox (i.e., after calling `modal.Sandbox.set_tags()`). Note that Sandbox tags are distinct from the new concept of App tags.
- `modal.Dict.pop()` now accepts an optional `default` parameter, akin to Python's `dict.pop()`.
- It's now possible to `modal shell` into a running Sandbox by passing its Sandbox ID (`modal shell sb-123`).
- Sandboxes can now be configured to expose a PTY device via `Sandbox.create(..., pty=True)` and `Sandbox.exec(..., pty=True)`. This provides better support for Claude Code.
- The new `modal.experimental.image_delete()` function can be used to delete the final layer of an Image given its ID, which can be particularly useful for cleaning up Sandbox Filesystem Snapshots.
- Using `modal run --interactive` (or `-i`) will now suppress Modal's status spinner to avoid interfering with breakpoints in local entrypoint functions. We've also improved support for printing large objects when attached to a debugger.
- We've improved support for Protobuf 5+ when using the Python implementation of the Protobuf runtime.

This release also introduces a small number of new deprecations:

- We deprecated the `client` parameter from `Sandbox.set_tags()`. To use an explicit Client when interacting with the Sandbox, pass it into `modal.Sandbox.create()` instead.
- We deprecated the `pty_info` parameter from `Sandbox.create()` and `Sandbox.exec()`. This was a private parameter accepting an internal Protobuf type. See the new boolean `pty` parameter instead.
- We replaced the `--no-confirm` option with `--yes` in the `modal environment delete` CLI to align with other CLI commands that normally require confirmation.

Finally, some functionality that began issuing deprecation warnings prior to v0.73 has now been completely removed:

- It is now required to "instantiate" a `modal.Cls` before invoking one of its methods.
- The eager `.lookup()` method has been removed from most Modal object classes (but not from `modal.App.lookup`, which remains supported). The lazy `.from_name()` method is recommended for accessing deployed objects going forward.
- The public constructors on the `modal.mount.Mount` object have been removed; this is now an entirely internal class.
- The `context_mount=` parameter has accordingly been removed from Docker-oriented `modal.Image` methods.
- The unused `allow_cross_region_volumes` parameter has been removed from the function decorators.
- The `modal.experimental.update_autoscaler()` function has been removed; this functionality now has a stable API as `modal.Function.update_autoscaler()`.

## 1.1

### 1.1.4 (2025-09-03)

- Added a `startup_timeout` parameter to the `@app.function()` and `@app.cls()` decorators. When used, this configures the timeout applied to each container's startup period separately from the input `timeout`. For backwards compatibility, `timeout` still applies to the startup phase when `startup_timeout` is unset.
- Added an optional `idle_timeout` parameter to `modal.Sandbox.create()`. When provided, Sandboxes will terminate after `idle_timeout` seconds of idleness.
- The dataclass returned by `modal.experimental.get_cluster_info()` now includes a `cluster_id` field to identify the clustered set of containers.
- When `block_network=True` is set in `modal.Sandbox.create()`, we now raise an error if any of `encrypted_ports`, `h2_ports`, or `unencrypted_ports` are also set.
- Functions decorated with `@modal.asgi_app()` now return an HTTP 408 (request timeout) error code instead of a 502 (gateway timeout) in rare cases when an input fails to arrive at the container, e.g. due to cancellation.
- `modal.Sandbox.create()` now warns when an invalid `name=` is passed, applying the same rules as other Modal object names: names must be alphanumeric and not longer than 64 characters. This will become an error in the future.

### 1.1.3 (2025-08-19)

- Fixed a bug introduced in `v1.1.2` that causes invocation of `modal.FunctionCall.get`, `modal.FunctionCall.get_call_graph`, `modal.FunctionCall.cancel`, and `modal.FunctionCall.gather` to fail when the `FunctionCall` object is retrieved via `modal.FunctionCall.from_id`.
- Added retries to improve the robustness of `modal volume get`

### 1.1.2 (2025-08-14)

We're introducing a new API pattern for imperative management of Modal resource types (`modal.Volume`, `modal.Secret`, `modal.Dict`, and `modal.Queue`). The API is accessible through the `.objects` namespace on each class. The object management namespace has methods for the following operations:

- `.objects.create(name)` creates an object on our backend. E.g., with [`modal.Volume.objects.create`](https://modal.com/docs/reference/modal.Volume#create):
  ```python notest
  modal.Volume.objects.create("huggingface-cache", environment_name="dev")
  ```
- `.objects.delete(name)` deletes the object with that name. E.g., with [`modal.Secret.objects.delete`](https://modal.com/docs/reference/modal.Secret#delete):
  ```python notest
  modal.Secret.objects.delete("aws-token")
  ```
- `.objects.list()` returns a list of object instances. E.g., with [`modal.Queue.objects.list`](https://modal.com/docs/reference/modal.Queue#list):
  ```python notest
  for queue in modal.Queue.objects.list():
      queue_info = queue.info()
      print(queue_info.name, queue_info.created_at, queue.len())
  ```

With the introduction of these APIs, we're replacing a few older methods with similar functionality:

- Static `.delete()` methods on the resource types themselves are being deprecated, because they are too easily confused with operations on the _contents_ of a resource (i.e., calling `modal.Dict.delete(key_name)` is an easy mistake that can have significant adverse consequences).
- The undocumented `.create_deployed()` methods of `modal.Volume` and `modal.Secret` are being deprecated in favor of this consistent API for imperative management.

Other changes:

- `modal.Cls.with_options` now supports `region` and `cloud` keyword arguments to support runtime constraints on scheduling.
- Fixed a bug that could cause Image builds to fail with `'FilePatternMatcher' object has no attribute 'patterns'` when using a `modal.FilePatternMatcher.from_file` ignore pattern.
- Fixed a bug where `rdma=True` was ignored when using `@modal.experimental.clustered()` with a `modal.Cls`.

### 1.1.1 (2025-08-01)

We're introducing the concept of "named Sandboxes" for usecases where Sandboxes need to have unique ownership over a resource. A named Sandbox can be created by passing `name=` to `modal.Sandbox.create()`, and it can be retrieved with the new `modal.Sandbox.from_name()` constructor. Only one running Sandbox can use a given name (scoped within the App that is managing the Sandbox) at any time, so trying to create a Sandbox with a name that is already taken will fail. Sandboxes release their name when they terminate. See the [guide](https://modal.com/docs/guide/sandbox#named-sandboxes) for more information about using this new feature.

Other changes:

- We've made an internal change to the `modal.Image.uv_pip_install` method to make it more portable across different base Images. As a consequence, Images built with this method on 1.1.0 will need to rebuild the next time they are used.
- We've added a `.name` property and `.info()` method to `modal.Dict`, `modal.Queue`, `modal.Volume`, and `modal.Secret` objects.
- Sandboxes now support `experimental_options` configuration for enabling preview functionality.
- We've Improved Modal's rich output when used in a Jupyter notebook.

### 1.1.0 (2025-07-17)

This release introduces support for the `2025.06` [Image Builder Version](https://modal.com/docs/guide/images#image-builder-updates), which is in a "preview" state. The new image builder includes several major changes to how the Modal client dependencies are included in Modal Images. These improvements should greatly reduce the risk of conflicts with user code dependencies. They also allow Modal Sandboxes to easily be used with existing Images or Dockerfiles that are not themselves compatible with the Modal client library. You can see more details and update your Workspace on its [Image Config](https://modal.com/settings/image-config) page. Please share any issues that you encounter as we work to make the version stable.

We're also introducing first-class support for building Modal Images with the [uv package manager](https://docs.astral.sh/uv/) through the new [`modal.Image.uv_pip_install`](https://modal.com/docs/reference/modal.Image#uv_pip_install) and [`modal.Image.uv_sync`](https://modal.com/docs/reference/modal.Image#uv_sync) methods:

```python
import modal

# uv_pip_install accepts a list of packages, like pip_install, but up to 50% faster
image = modal.Image.debian_slim().uv_pip_install("torch==2.7.1", "numpy==2.3.1")

# uv_sync accepts a local `uv_project_dir` (defaulting to the local working directory)
# and uses the pyproject.toml and uv.lock files to specify the environment
image = modal.Image.debian_slim().uv_sync()
```

Please note that, as these methods are new, there is some chance that future releases will need to fix bugs or address edge cases in ways that break the cache for existing Images. When using `modal.Image.uv_pip_install`, we recommend pinning dependency versions so that any necessary rebuilds produce a consistent environment.

This release also includes a number of other new features and bug fixes:

- Optimized handling of the `ignore` parameter in `Image.add_local_dir` and similar methods for cases where entire directories are ignored.
- Added a `poetry_version` parameter to `modal.Image.poetry_install_from_file`, which supports installing a specific version of `poetry`. It's also possible to set `poetry_version=None` to skip the install step, i.e. when poetry is already available in the Image.
- Added a [`modal.Sandbox.reload_volumes`](https://modal.com/docs/reference/modal.Sandbox#reload_volumes) method, which triggers a reload of all Volumes currently mounted inside a running Sandbox.
- Added a `build_args` parameter to `modal.Image.from_dockerfile` for passing arguments through to `ARG` instructions in the Dockerfile.
- It's now possible to use `@modal.experimental.clustered` and `i6pn` networking with `modal.Cls`.
- Fixed a bug where `Cls.with_options` would fail when provided with a `modal.Secret` object that was already hydrated.
- Fixed a bug where the timeout specified in `modal.Sandbox.exec()` was not respected by `ContainerProcess.wait()` or `ContainerProcess.poll()`.
- Fixed retry handling when using `modal run --detach` directly against a remote Function.

Finally, this release introduces a small number of deprecations and potentially-breaking changes:

- We now raise `modal.exception.NotFoundError` in all cases where Modal object lookups fail; previously some methods could leak an internal `GRPCError` with a `NOT_FOUND` status.
- We're enforcing pre-1.0 deprecations on `modal.build`, `modal.Image.copy_local_file`, and `modal.Image.copy_local_dir`.
- We're deprecating the `environment_name` parameter in `modal.Sandbox.create()`. A Sandbox's environment association will now be determined by its parent App. This should have no user-facing effects.
- We've deprecated the `namespace` parameter in the `.from_name` methods of `Function`, `Cls`, `Dict`, `Queue`, `Volume`, `NetworkFileSystem`, and `Secret`, along with `modal.runner.deploy_app`. These object types do not have a concept of distinct namespaces.

## 1.0

### 1.0.5 (2025-06-27)

- Added a [`modal.Volume.read_only`](/docs/reference/modal.Volume#read_only) method, which will configure a Volume instance to disallow writes:

  ```python notest
  vol = modal.Volume.from_name("models")
  read_only_vol = vol.read_only()

  @app.function(volumes={"/models": read_only_vol})
  def f():
      with open("/models/weights.pt", "w") as fid:  # Raises an OSError
          ...

  @app.local_entrypoint()
  def main():
      with read_only_vol.batch_upload() as batch:  # Raises a modal.exceptions.InvalidError
          ...

      with vol.batch_upload() as batch:  # This instance is still writeable
          ...
  ```

- Introduced a gradual fix for a bug where `Function.map` and `Function.starmap` leak an internal exception wrapping type (`modal.exceptions.UserCodeException`) when `return_exceptions=True` is set. To avoid breaking any user code that depends on the specific types in the return list, these functions will continue returning the wrapper type by default, but they now issue a deprecation warning. To opt into the future behavior and silence the warning, you can set `wrap_returned_exceptions=False` in the call to `.map` or `.starmap`.
- When an `@app.cls()`-decorated class inherits from a class (or classes) with `modal.parameter()` annotations, the parent parameters will now be inherited and included in the parameter set for the modal Cls.
- Redeployments that migrate parameterized functions from an explicit constructor to `modal.parameter()` annotations will now handle requests from outdated clients more gracefully, avoiding a problem where new containers would crashloop on a deserialization error.
- The Modal client will now retry its initial connection to the Modal server, improving stability on flaky networks.

### 1.0.4 (2025-06-13)

- When `modal.Cls.with_options` is called multiple times on the same instance, the overrides will now be merged. For example, the following configuration will use an H100 GPU and request 16 CPU cores:
  ```python
  Model.with_options(gpu="A100", cpu=16).with_options(gpu="H100")
  ```
- Added a `--secret` option to `modal shell` for including environment variables defined by named Secret(s) in the shell session:
  ```
  modal shell --secret huggingface --secret wandb
  ```
- Added a `verbose: bool` option to `modal.Sandbox.create()`. When this is set to `True`, execs and file system operations will appear in the Sandbox logs.
- Updated `modal.Sandbox.watch()` so that exceptions are now raised in (and can be caught by) the calling task.

### 1.0.3 (2025-06-05)

- Added support for specifying a timezone on `Cron` schedules, which allows you to run a Function at a specific local time regardless of daylight savings:

  ```python
  import modal
  app = modal.App()

  @app.function(schedule=modal.Cron("* 6 * * *"), timezone="America/New_York")  # Use tz database naming conventions
  def f():
      print("This function will run every day at 6am New York time.")
  ```

- Added an `h2_ports` parameter to `Sandbox.create`, which exposes encrypted ports using HTTP/2. The following example will create an H2 port on 5002 and a port using HTTPS over HTTP/1.1 on 5003:
  ```python
  sb = modal.Sandbox.create(app=app, h2_ports = [5002], encrypted_ports = [5003])
  ```
- Added `--from-dotenv` and `--from-json` options to `modal secret create`, which will read from local files to populate Secret contents.
- `Sandbox.terminate` no longer waits for container shutdown to complete before returning. It still ensures that a terminated container will shutdown imminently. To restore the previous behavior (i.e., to wait until the Sandbox is actually terminated), call `sb.wait(raise_on_termination=False)` after calling `sb.terminate()`.
- Improved performance and stability for `modal volume get`.
- Fixed a rare race condition that could sometimes make `Function.map` and similar calls deadlock.
- Fixed an issue where `Function.map` and similar methods would stall for 55 seconds when passed an empty iterator as input instead of completing immediately.
- We now raise an error during App setup when using interactive mode without the `modal.enable_output` context manager. Previously, this would run the App but raise when `modal.interact()` was called.

### 1.0.2 (2025-05-26)

- Fixed an incompatibility with breaking changes in `aiohttp` v3.12.0, which caused issues with Volume and large input uploads. The issues typically manifest as `Local data and remote data checksum mismatch` or `'_io.BufferedReader' object has no attribute 'getbuffer'` errors.

### 1.0.1 (2025-05-19)

- Added a `--timestamps` flag to `modal app logs` that prepends a timestamp to each log line.
- Fixed a bug where objects returned by `Sandbox.list` had `returncode == 0` for _running_ Sandboxes. Now the return code for running Sandboxes will be `None`.
- Fixed a bug affecting systems where the `sys.platform.node` name includes unicode characters.

### 1.0.0 (2025-05-16)

With this release, we're beginning to enforce the deprecations discussed in the [1.0 migration guide](https://modal.com/docs/guide/modal-1-0-migration). Going forward, we'll include breaking changes for outstanding deprecations in `1.Y.0` releases, so we recommend pinning Modal on a minor version (`modal~=1.0.0`) if you have not addressed the existing warnings. While we'll continue to make improvements to the Modal API, new deprecations will be introduced at a substantially reduced rate, and support windows for older client versions will lengthen.

⚠️ In this release, we've made some breaking changes to Modal's "automounting" behavior.️ If you've not already adapted your source code in response to warnings about automounting, Apps built with 1.0+ will have different files included and may not run as expected:

- Previously, Modal containers would automatically include the source for local Python packages that were imported by your Modal App. Going forward, it will be necessary to explicitly include such packages in the Image (i.e., with `modal.Image.add_local_python_source`).
- Support for the `automount` configuration (`MODAL_AUTOMOUNT`) has been removed; this environment variable will no longer have any effect.
- Modal will continue to automatically include the Python module or package where the Function is defined. If the Function is defined within a package, the entire directory tree containing the package will be mounted. This limited automounting can also be disabled in cases where your Image definition already includes the package defining the Function: set `include_source=False` in the `modal.App` constructor or `@app.function` decorator.

Additionally, we have enforced a number of previously-introduced deprecations:

- Removed `modal.Mount` as a public object, along with various `mount=` parameters where Mounts could be passed into the Modal API. Usage can be replaced with `modal.Image` methods, e.g.:
  ```python
  @app.function(image=image, mounts=[modal.Mount.from_local_dir("data", "/root/data")])  # This is now an error!
  @app.function(image=image.add_local_dir("data", "/root/data"))  # Correct spelling
  ```
- Removed the `show_progress` parameter from `modal.App.run`. This parameter has been replaced by the `modal.enable_output` context manager:
  ```python
  with modal.enable_output(), app.run():
    ...  # Will produce verbose Modal output
  ```
- Passing flagged options to the `Image.pip_install` package list will now raise an error. Use the `extra_options` parameter to specify options that aren't exposed through the `Image.pip_install` signature:
  ```python
  image.pip_install("flash-attn", "--no-build-isolation")  # This is now an error!
  image.pip_install("flash-attn", extra_options="--no-build-isolation")  # Correct spelling
  ```
- Removed backwards compatibility for using `label=` or `tag=` keywords in object lookup methods. We standardized these methods to use `name=` as the parameter name, but we recommend using positional arguments:
  ```python
  f = modal.Function.from_name("my-app", tag="f")  # No longer supported! Will raise an error!
  f = modal.Function.from_name("my-app", "f")  # Preferred spelling
  ```
- It's no longer possible to invoke a generator Function with `Function.spawn`; previously this warned, now it raises an `InvalidError`. Additionally, the `FunctionCall.get_gen` method has been removed, and it's no longer possible to set `is_generator` when using `FunctionCall.from_id`.
- Removed the `.resolve()` method on Modal objects. This method had not been publicly documented, but where used it can be replaced straightforwardly with `.hydrate()`. Note that explicit hydration should rarely be necessary: in most cases you can rely on lazy hydration semantics (i.e., objects will be hydrated when the first method that requires server metadata is called).
- Functions decorated with `@modal.asgi_app` or `@modal.wsgi_app` are now required to be nullary. Previously, we warned in the case where a function was defined with parameters that all had default arguments.
- Referencing the deprecated `modal.Stub` object will now raise an `AttributeError`, whereas previously it was an alias for `modal.App`. This is a simple name change.
