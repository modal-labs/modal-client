# Changelog

This changelog documents user-facing updates (features, enhancements, fixes, and deprecations) to the `modal` client library.

## Latest

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

## 0.77

### 0.77.0 (2025-05-13)

- This is the final pre-1.0 release of the Modal client. The next release will be version 1.0. While we do not plan to enforce most major deprecations until later in the 1.0 cycle, there will be some breaking changes introduced in the next release.

## 0.76

### 0.76.3 (2025-05-12)

- Fixed the behavior of `modal app history --json` when the history contains versions with and without commit information or "tag" metadata. Commit information is now always included (with a `null` placeholder when absent), while tag metadata is included only when there is at least one tagged release (other releases will have a `null` placeholder).

### 0.76.0 (2025-05-12)

- Fixed the behavior of `ignore=` in `modal.Image` methods, including when `.dockerignore` files are implicitly used in docker-oriented methods. This may result in Image rebuilds with different final inventories:
  - When using `modal.Image.add_local_dir`, exclusion patterns are now correctly interpreted as relative to the directory being added (e.g., `*.json` will now ignore all json files in the top-level of the directory).
  - When using `modal.Image.from_dockerfile`, exclusion patterns are correctly interpreted as relative to the context directory.
  - As in Docker, leading or trailing path delimiters are stripped from the ignore patterns before being applied.
  - **Breaking change**: When providing a custom function to `ignore=`, file paths passed into the function will now be _relative_, rather than absolute.

## 0.75

### 0.75.8 (2025-05-12)

- Introduced `modal.Cls.with_concurrency` and `modal.Cls.with_batching` for runtime configuration of functionality that is exposed through the `@modal.concurrent` and `@modal.batched` decorators.
  ```python
  model = Model.with_options(gpu="H100").with_concurrency(max_inputs=100)()
  ```
- Added a deprecation warning when using `allow_concurrent_inputs` in `modal.Cls.with_options`.
- Added `buffer_containers` to `modal.Cls.with_options`.
- _Behavior change:_ when `modal.Cls.with_options` is called multiple times on the same object, the configurations will be merged rather than using the most recent.

### 0.75.4 (2025-05-09)

- Fixed issue with .spawn_map producing wrong number of arguments

### 0.75.3 (2025-05-08)

- New `modal.Dict`s (forthcoming on 2025-05-20) use a new durable storage backend with more "cache-like" behavior - items expire after 7 days of inactivity (no reads or writes). Previously created `modal.Dict`s will continue to use the old backend, but support will eventually be dropped.
- `modal.Dict.put` now supports an `skip_if_exists` flag that can be used to avoid overwriting the value for existing keys:

  ```
  item_created = my_dict.put("foo", "bar", skip_if_exists=True)
  assert item_created
  new_item_created = my_dict.put("foo", "baz", skip_if_exists=True)
  assert not new_item_created
  ```

  Note that this flag only works for `modal.Dict` objects with the new backend (forthcoming on 2025-05-20) and will raise an error otherwise.

### 0.75.2 (2025-05-08)

- Reverts defective changes to the interpretation of `ignore=` patterns and `.dockerignore` files that were introduced in v0.75.0.

### 0.75.0 (2025-05-08)

- Introduced some changes to the handling of `ignore=` patterns in `modal.Image` methods. Due to a defect around the handling of leading path delimiter characters, these changes reverted in 0.75.2 and later reintroduced in 0.76.0.

## 0.74

### 0.74.63 (2025-05-08)

- Deprecates `Function.web_url` in favor of a new `Function.get_web_url()` method. This also allows the url of a `Function` to be retrieved in an async manner using `Function.get_web_url.aio()` (like all other io-bearing methods in the Modal API)

### 0.74.61 (2025-05-07)

- Adds a deprecation warning when data is passed directly to `modal.Dict.from_name` or `modal.Dict.ephemeral`. Going forward, it will be necessary to separate `Dict` population from creation.

### 0.74.60 (2025-05-07)

- `modal.Dict.update` now also accepts a positional Mapping, like Python's `dict` type:

  ```python
  d = modal.Dict.from_name("some-dict")
  d.update({"a_key": 1, "another_key": "b"}, some_kwarg=True)
  ```

### 0.74.56 (2025-05-06)

- Experimental `modal cluster` subcommand is added.

### 0.74.53 (2025-05-06)

- Added functionality for `.spawn_map` on a function instantiated from `Function.from_name`.

### 0.74.51 (2025-05-06)

- The `modal` client library can now be installed with Protobuf 6.

### 0.74.49 (2025-05-06)

- Changes the log format of the modal client's default logger. Instead of `[%(threadName)s]`, the client now logs `[modal-client]` as the log line prefix.
- Adds a configuration option (MODAL_LOG_PATTERN) to the modal config for setting the log formatting pattern, in case users want to customize the format. To get the old format, use `MODAL_LOG_PATTERN='[%(threadName)s] %(asctime)s %(message)s'` (or add this to your `.modal.toml` in the `log_pattern` field).

### 0.74.48 (2025-05-05)

- Added a new method for spawning many function calls in parallel: `Function.spawn_map`.

### 0.74.46 (2025-05-05)

- Introduces a new `.update_autoscaler()` method, which will replace the existing `.keep_warm()` method with the ability to dynamically change the entire autoscaler configuration (`min_containers`, `max_containers`, `buffer_containers`, and `scaledown_window`).

### 0.74.39 (2025-04-30)

- The `modal` client no longer includes `fastapi` as a library dependency.

### 0.74.36 (2025-04-29)

- A new parameter, `restrict_modal_access`, can be provided on a Function to prevent it from interacting with other resources in your Modal Workspace like Queues, Volumes, or other Functions. This can be useful for running user-provided or LLM-written code in a safe way.

### 0.74.35 (2025-04-29)

- Fixed a bug that prevented doing `modal run` against an entrypoint defined by `Cls.with_options`.

### 0.74.32 (2025-04-29)

- When setting a custom `name=` in `@app.function()`, an error is now raised unless `serialized=True` is also set.

### 0.74.25 (2025-04-25)

- The `App.include` method now returns `self` so it's possible to build up an App through chained calls:

  ```python
  app = modal.App("main-app").include(sub_app_1).include(sub_app_2)
  ```

### 0.74.23 (2025-04-25)

- Marked some parameters in a small number of Modal functions as requiring keyword arguments (namely, `modal.App.run`, `modal.Cls.with_options`, all `.from_name` methods, and a few others). Code that calls these functions with positional arguments will now raise an error. This is expected to be minimally disruptive as the affected parameters are mostly "extra" options or positioned after parameters that have previously been deprecated.

### 0.74.22 (2025-04-24)

- Added a `modal secret delete` command to the CLI.

### 0.74.21 (2025-04-24)

- The `allow_cross_region_volumes` parameter of the `@app.function` and `@app.cls` decorators now issues a deprecation warning; the parameter is always treated as `True` on the Modal backend.

### 0.74.18 (2025-04-23)

- Adds a `.deploy()` method to the `App` object. This method allows you programmatically deploy Apps from Python:

  ```python
  app = modal.App("programmatic-deploy")
  ...
  app.deploy()
  ```

### 0.74.12 (2025-04-18)

- The `@app.function` and `@app.cls` decorators now support `experimental_options`, which we'll use going forward when testing experimental functionality that depends only on server-side configuration.

### 0.74.7 (2025-04-17)

- Modal will now raise an error if local files included in the App are modified during the build process. This behavior can be controlled with the `MODAL_BUILD_VALIDATION` configuration, which accepts `error` (default), `warn`, or `ignore`.

### 0.74.6 (2025-04-17)

- Internal change that makes containers for functions/classes with `serialized=True` start up _slightly_ faster than before

### 0.74.0 (2025-04-15)

- Introduces a deprecation warning when using explicit constructors (`__init__` methods) on `@modal.cls`-decorated classes. Class parameterization should instead be done via [dataclass-style `modal.parameter()` declarations](https://modal.com/docs/guide/parametrized-functions). Initialization logic should run in `@modal.enter()`-decorated [lifecycle methods](https://modal.com/docs/guide/lifecycle-functions).

## 0.73

### 0.73.173 (2025-04-15)

- Fix bug where containers hang with batch sizes above 100 (with `@modal.batched`).
- Fix bug where containers can fail with large outputs and batch sizes above 49 (with `@modal.batched`)

### 0.73.170 (2025-04-14)

- Fixes a bug where `modal run` didn't recognize `modal.parameter()` class parameters

### 0.73.165 (2025-04-11)

- Allow running new ephemeral apps from **within** Modal containers using `with app.run(): ...`. Use with care, as putting such a run block in global scope of a module could easily lead to infinite app creation recursion

### 0.73.160 (2025-04-10)

- The `allow_concurrent_inputs` parameter of `@app.function` and `@app.cls` is now deprecated in favor of the `@modal.concurrent` decorator. See the [Modal 1.0 Migration Guide](https://modal.com/docs/guide/modal-1-0-migration#replacing-allow_concurrent_inputs-with-modalconcurrent) and documentation on [input concurrency](https://modal.com/docs/guide/concurrent-inputs) for more information.

### 0.73.159 (2025-04-10)

- Fixes a bug where `serialized=True` classes could not `self.` reference other methods on the class, or use `modal.parameter()` synthetic constructors

### 0.73.158 (2025-04-10)

- Adds support for `bool` type to class parameters using `name: bool = modal.parameter()`. Note that older clients can't instantiate classes with bool parameters unless those have default values which are not modified. Bool parameters are also not supported by web endpoints at this time.

### 0.73.148 (2025-04-07)

- Fixes a bug introduced in 0.73.147 that broke App builds when using `@modal.batched` on a class method.

### 0.73.147 (2025-04-07)

- Improved handling of cases where `@modal.concurrent` is stacked with other decorators.

### 0.73.144 (2025-04-04)

- Adds a `context_dir` parameter to `modal.Image.from_dockerfile` and `modal.Image.dockerfile_commands`. This parameter can be used to provide a local reference for relative COPY commands.

### 0.73.139 (2025-04-02)

- Added `modal.experimental.ipython` module, which can be loaded in Jupyter notebooks with `%load_ext modal.experimental.ipython`. Currently it provides the `%modal` line magic for looking up functions:

  ```python
  %modal from main/my-app import my_function, MyClass as Foo

  # Now you can use my_function() and Foo in your notebook.
  my_function.remote()
  Foo().my_method.remote()
  ```

- Removed the legacy `modal.extensions.ipython` module from 2022.

### 0.73.135 (2025-03-29)

- Fix shutdown race bug that emitted spurious error-level logs.

### 0.73.132 (2025-03-28)

- Adds the `@modal.concurrent` decorator, which will be replacing the beta `allow_concurrent_inputs=` parameter of `@app.function` and `@app.cls` for enabling input concurrency. Notably, `@modal.concurrent` introduces a distinction between `max_inputs` and `target_inputs`, allowing containers to burst over the concurrency level targeted by the Modal autoscaler during periods of high load.

### 0.73.131 (2025-03-28)

- Instantiation of classes using keyword arguments that are not defined as as `modal.parameter()` will now raise an error on the calling side rather than in the receiving container. Note that this only applies if there is at least one modal.parameter() defined on the class, but this will likely apply to parameter-less classes in the future as well.

### 0.73.121 (2025-03-24)

- Adds a new "commit info" column to the `modal app history` command. It shows the short git hash at the time of deployment, with an asterisk `*` if the repository had uncommitted changes.

### 0.73.119 (2025-03-21)

- Class parameters are no longer automatically cast into their declared type. If the wrong type is provided to a class parameter, method calls to that class instance will now fail with an exception.

### 0.73.115 (2025-03-19)

- Adds support for new strict `bytes` type for `modal.parameter`

Usage:

```py
import typing
import modal

app = modal.App()


@app.cls()
class Foo:
    a: bytes = modal.parameter(default=b"hello")

    @modal.method()
    def bar(self):
        return f"hello {self.a}"


@app.local_entrypoint()
def main():
    foo = Foo(a=b"world")
    foo.bar.remote()
```

**Note**: For parameterized web endoints you must base64 encode the bytes before passing them in as a query parameter.

### 0.73.107 (2025-03-14)

- Include git commit info at the time of app deployment.

### 0.73.105 (2025-03-14)

- Added `Image.cmd()` for setting image default entrypoint args (a.k.a. `CMD`).

### 0.73.95 (2025-03-12)

- Fixes a bug which could cause `Function.map` and sibling methods to stall indefinitely if there was an exception in the input iterator itself (i.e. not the mapper function)

### 0.73.89 (2025-03-05)

- The `@modal.web_endpoint` decorator is now deprecated. We are replacing it with `@modal.fastapi_endpoint`. This can be a simple name substitution in your code; the two decorators have identical semantics.

### 0.73.84 (2025-03-04)

- The `keep_warm=` parameter has been removed from the`@modal.method` decorator. This parameter has been nonfunctional since v0.63.0; all autoscaler configuration must be done at the level of the modal Cls.

### 0.73.82 (2025-03-04)

- Adds `modal.fastapi_endpoint` as an alias for `modal.web_endpoint`. We will be deprecating the `modal.web_endpoint` _name_ (but not the functionality) as part of the Modal 1.0 release.

### 0.73.81 (2025-03-03)

- The `wait_for_response` parameter of Modal's web endpoint decorators has been removed (originally deprecated in May 2024).

### 0.73.78 (2025-03-01)

- It is now possible to call `Cls.with_options` on an unhydrated Cls, e.g.

  ```python
  ModelWithGPU = modal.Cls.from_name("my-app", "Model").with_options(gpu="H100")
  ```

### 0.73.77 (2025-03-01)

- `Cls.with_options()` now accept unhydated volume and secrets

### 0.73.76 (2025-02-28)

- We're renaming several `App.function` and `App.cls` parameters that configure the behavior of Modal's autoscaler:
  - `concurrency_limit` is now `max_containers`
  - `keep_warm` is now `min_containers`
  - `container_idle_timeout` is now `scaledown_window`
- The old names will continue to work, but using them will issue a deprecation warning. The aim of the renaming is to reduce some persistent confusion about what these parameters mean. Code updates should require only a simple substitution of the new name.
- We're adding a new parameter, `buffer_containers` (previously available as `_experimental_buffer_containers`). When your Function is actively handling inputs, the autoscaler will spin up additional `buffer_containers` so that subsequent inputs will not be blocked on cold starts. When the Function is idle, it will still scale down to the value given by `min_containers`.

### 0.73.75 (2025-02-28)

- Adds a new config field, `ignore_cache` (also accessible via environment variables as `MODAL_IGNORE_CACHE=1`), which will force Images used by the App to rebuild but not clobber any existing cached Images. This can be useful for testing an App's robustness to Image rebuilds without affecting other Apps that depend on the same base Image layer(s).

### 0.73.73 (2025-02-28)

- Adds a deprecation warning to the `workspace` parameter in `modal.Cls` lookup methods. This argument is unused and will be removed in the future.

### 0.73.69 (2025-02-25)

- We've moved the `modal.functions.gather` function to be a staticmethod on `modal.FunctionCall.gather`. The former spelling has been deprecated and will be removed in a future version.

### 0.73.68 (2025-02-25)

- Fixes issue where running `modal shell` with a dot-separated module reference as input would not accept the required `-m` flag for "module mode", but still emitted a warning telling users to use `-m`

### 0.73.60 (2025-02-20)

- Fixes an issue where `modal.runner.deploy_app()` didn't work when called from within a running (remote) Modal Function

### 0.73.58 (2025-02-20)

- Introduces an `-m` flag to `modal run`, `modal shell`, `modal serve` and `modal deploy`, which indicates that the modal app/function file is specified using python "module syntax" rather than a file path. In the future this will be a required flag when using module syntax.

  Old syntax:

  ```sh
  modal run my_package/modal_main.py
  modal run my_package.modal_main
  ```

  New syntax (note the `-m` on the second line):

  ```sh
  modal run my_package/modal_main.py
  modal run -m my_package.modal_main
  ```

### 0.73.54 (2025-02-18)

- Passing `App.lookup` an invalid name now raises an error. App names may contain only alphanumeric characters, dashes, periods, and underscores, must be shorter than 64 characters, and cannot conflict with App ID strings.

### 0.73.51 (2025-02-14)

- Fixes a bug where sandboxes returned from `Sandbox.list()` were not snapshottable even if they were created with `_experimental_enable_snapshot`.

### 0.73.44 (2025-02-13)

- `modal.FunctionCall` is now available in the top-level `modal` namespace. We recommend referencing the class this way instead of using the the fully-qualified `modal.functions.FunctionCall` name.

### 0.73.40 (2025-02-12)

- `Function.web_url` will now return None (instead of raising an error) when the Function is not a web endpoint

### 0.73.31 (2025-02-10)

- Deprecate the GPU classes (`gpu=A100(...)` etc) in favor of just using strings (`gpu="A100"` etc)

### 0.73.26 (2025-02-10)

- Adds a pending deprecation warning when looking up class methods using `Function.from_name`, e.g. `Function.from_name("some_app", "SomeClass.some_method")`. The recommended way to reference methods of classes is to look up the class instead: `RemoteClass = Cls.from_name("some_app", "SomeClass")`

### 0.73.25 (2025-02-09)

- Fixes an issue introduced in `0.73.19` that prevented access to GPUs during image builds

### 0.73.18 (2025-02-06)

- When using a parameterized class (with at least one `modal.parameter()` specified), class instantiation with an incorrect construction signature (wrong arguments or types) will now fail at the `.remote()` calling site instead of container startup for the called class.

### 0.73.14 (2025-02-04)

- Fixed the status message shown in terminal logs for ephemeral Apps to accurately report the number of active containers.

### 0.73.11 (2025-02-04)

- Warns users if the `modal.Image` of a Function/Cls doesn't include all the globally imported "local" modules (using `.add_local_python_source()`), and the user hasn't explicitly set an `include_source` value of True/False. This is in preparation for an upcoming deprecation of the current "auto mount" logic.

### 0.73.10 (2025-02-04)

- Modal functions, methods and entrypoints can now accept variable-length arguments to skip Modal's default CLI parsing. This is useful if you want to use Modal with custom argument parsing via `argparse` or `HfArgumentParser`. For example, the following function can be invoked with `modal run my_file.py --foo=42 --bar="baz"`:

  ```python
  import argparse

  @app.function()
  def train(*arglist):
      parser = argparse.ArgumentParser()
      parser.add_argument("--foo", type=int)
      parser.add_argument("--bar", type=str)
      args = parser.parse_args(args = arglist)
  ```

### 0.73.1 (2025-01-30)

- `modal run` now runs a single local entrypoints/function in the selected module. If exactly one local entrypoint or function exists in the selected module, the user doesn't have to qualify the runnable
  in the modal run command, even if some of the module's referenced apps have additional local entrypoints or functions. This partially restores "auto-inferred function" functionality that was changed in v0.72.48.

### 0.73.0 (2025-01-30)

- Introduces an `include_source` argument in the `App.function` and `App.cls` decorators that let users configure which class of python packages are automatically included as source mounts in created modal functions/classes (what we used to call "automount" behavior). This will supersede the MODAL_AUTOMOUNT configuration value which will eventually be deprecated. As a convenience, the `modal.App` constructor will also accept an `include_source` argument which serves as the default for all the app's functions and classes.

  The `include_source` argument accepts the following values:

  - `True` (default in a future version of Modal) Automatically includes the Python files of the source package of the function's own home module, but not any other local packages. Roughly equivalent ot `MODAL_AUTOMOUNT=0` in previous versions of Modal.
  - `False` - don't include _any_ local source. Assumes the function's home module is importable in the container environment through some other means (typically added to the provided `modal.Image`'s Python environment).
  - `None` (the default) - use current soon-to-be-deprecated automounting behavior, including source of all first party packages that are not installed into site-packages locally.

- Minor change to `MODAL_AUTOMOUNT=0`: When running/deploying using a module path (e.g. `modal run mypak.mymod`), **all non .pyc files** of the source package (`mypak` in this case) are now included in the function's container. Previously, only the function's home `.py` module file + any `__init__.py` files in its package structure were included. Note that this is only for MODAL_AUTOMOUNT=0. To get full control over which source files are included with your functions, you can set `include_source=False` on your function (see above) and manually specify the files to include using the `ignore` argument to `Image.add_local_python_source`.

## 0.72

### 0.72.56 (2025-01-28)

- Deprecated `.lookup` methods on Modal objects. Users are encouraged to use `.from_name` instead. In most cases this will be a simple name substitution. See [the 1.0 migration guide](https://modal.com/docs/guide/modal-1-0-migration#deprecating-the-lookup-method-on-modal-objects) for more information.

### 0.72.54 (2025-01-28)

- Fixes bug introduced in v0.72.48 where `modal run` didn't work with files having global `Function.from_name()`/`Function.lookup()`/`Cls.from_name()`/`Cls.lookup()` calls.

### 0.72.48 (2025-01-24)

- Fixes a CLI bug where you couldn't reference functions via a qualified app, e.g. `mymodule::{app_variable}.{function_name}`.
- The `modal run`, `modal serve` and `modal shell` commands get more consistent error messages in cases where the passed app or function reference isn't resolvable to something that the current command expects.
- Removes the deprecated `__getattr__`, `__setattr__`, `__getitem__` and `__setitem__` methods from `modal.App`

### 0.72.39 (2025-01-22)

- Introduced a new public method, `.hydrate`, for on-demand hydration of Modal objects. This method replaces the existing semi-public `.resolve` method, which is now deprecated.

### 0.72.33 (2025-01-20)

- The Image returned by `Sandbox.snapshot_filesystem` now has `object_id` and other metadata pre-assigned rather than require loading by subsequent calls to sandboxes or similar to set this data.

### 0.72.30 (2025-01-18)

- Adds a new `oidc_auth_role_arn` field to `CloudBucketMount` for using OIDC authentication to create the mountpoint.

### 0.72.24 (2025-01-17)

- No longer prints a warning if `app.include` re-includes an already included function (warning is still printed if _another_ function with the same name is included)

### 0.72.22 (2025-01-17)

- Internal refactor of the `modal.object` module. All entities except `Object` from that module have now been moved to the `modal._object` "private" module.

### 0.72.17 (2025-01-16)

- The `@modal.build` decorator is now deprecated. For storing large assets (e.g. model weights), we now recommend using a `modal.Volume` over writing data to the `modal.Image` filesystem directly.

### 0.72.16 (2025-01-16)

- Fixes bug introduced in v0.72.9 where `modal run SomeClass.some_method` would incorrectly print a deprecation warning.

### 0.72.15 (2025-01-15)

- Added an `environment_name` parameter to the `App.run` context manager.

### 0.72.8 (2025-01-10)

- Fixes a bug introduced in v0.72.2 when specifying `add_python="3.9"` in `Image.from_registry`.

### 0.72.0 (2025-01-09)

- The default behavior`Image.from_dockerfile()` and `image.dockerfile_commands()` if no parameter is passed to `ignore` will be to automatically detect if there is a valid dockerignore file in the current working directory or next to the dockerfile following the same rules as `dockerignore` does using `docker` commands. Previously no patterns were ignored.

## 0.71

### 0.71.13 (2025-01-09)

- `FilePatternMatcher` has a new constructor `from_file` which allows you to read file matching patterns from a file instead of having to pass them in directly, this can be used for `Image` methods accepting an `ignore` parameter in order to read ignore patterns from files.

### 0.71.11 (2025-01-08)

- Modal Volumes can now be renamed via the CLI (`modal volume rename`) or SDK (`modal.Volume.rename`).

### 0.71.7 (2025-01-08)

- Adds `Image.from_id`, which returns an `Image` object from an existing image id.

### 0.71.1 (2025-01-06)

- Sandboxes now support fsnotify-like file watching:

```python
from modal.file_io import FileWatchEventType

app = modal.App.lookup("file-watch", create_if_missing=True)
sb = modal.Sandbox.create(app=app)
events = sb.watch("/foo")
for event in events:
    if event.type == FileWatchEventType.Modify:
        print(event.paths)
```

## 0.70

### 0.70.1 (2024-12-27)

- The sandbox filesystem API now accepts write payloads of sizes up to 1 GiB.

## 0.69

### 0.69.0 (2024-12-21)

- `Image.from_dockerfile()` and `image.dockerfile_commands()` now auto-infer which files need to be uploaded based on COPY commands in the source if `context_mount` is omitted. The `ignore=` argument to these methods can be used to selectively omit files using a set of glob patterns.

## 0.68

### 0.68.53 (2024-12-20)

- You can now point `modal launch vscode` at an arbitrary Dockerhub base image:

  `modal launch vscode --image=nvidia/cuda:12.4.0-devel-ubuntu22.04`

### 0.68.44 (2024-12-19)

- You can now run GPU workloads on [Nvidia L40S GPUs](https://www.nvidia.com/en-us/data-center/l40s/):

  ```python
  @app.function(gpu="L40S")
  def my_gpu_fn():
      ...
  ```

### 0.68.43 (2024-12-19)

- Fixed a bug introduced in v0.68.39 that changed the exception type raise when the target object for `.from_name`/`.lookup` methods was not found.

### 0.68.39 (2024-12-18)

- Standardized terminology in `.from_name`/`.lookup`/`.delete` methods to use `name` consistently where `label` and `tag` were used interchangeably before. Code that invokes these methods using `label=` as an explicit keyword argument will issue a deprecation warning and will break in a future release.

### 0.68.29 (2024-12-17)

- The internal `deprecation_error` and `deprecation_warning` utilities have been moved to a private namespace

### 0.68.28 (2024-12-17)

- Sandboxes now support additional filesystem commands `mkdir`, `rm`, and `ls`.

```python
app = modal.App.lookup("sandbox-fs", create_if_missing=True)
sb = modal.Sandbox.create(app=app)
sb.mkdir("/foo")
with sb.open("/foo/bar.txt", "w") as f:
    f.write("baz")
print(sb.ls("/foo"))
```

### 0.68.27 (2024-12-17)

- Two previously-introduced deprecations are now enforced and raise an error:
  - The `App.spawn_sandbox` method has been removed in favor of `Sandbox.create`
  - `Sandbox.create` now requires an `App` object to be passed

### 0.68.24 (2024-12-16)

- The `modal run` CLI now has a `--write-result` option. When you pass a filename, Modal will write the return value of the entrypoint function to that location on your local filesystem. The return value of the function must be either `str` or `bytes` to use this option; otherwise, an error will be raised. It can be useful for exercising a remote function that returns text, image data, etc.

### 0.68.21 (2024-12-13)

Adds an `ignore` parameter to our `Image` `add_local_dir` and `copy_local_dir` methods. It is similar to the `condition` method on `Mount` methods but instead operates on a `Path` object. It takes either a list of string patterns to ignore which follows the `dockerignore` syntax implemented in our `FilePatternMatcher` class, or you can pass in a callable which allows for more flexible selection of files.

Usage:

```python
img.add_local_dir(
  "./local-dir",
  remote_path="/remote-path",
  ignore=FilePatternMatcher("**/*", "!*.txt") # ignore everything except files ending with .txt
)

img.add_local_dir(
  ...,
  ignore=~FilePatternMatcher("**/*.py") # can be inverted for when inclusion filters are simpler to write
)

img.add_local_dir(
  ...,
  ignore=["**/*.py", "!module/*.py"] # ignore all .py files except those in the module directory
)

img.add_local_dir(
  ...,
  ignore=lambda fp: fp.is_relative_to("somewhere") # use a custom callable
)
```

which will add the `./local-dir` directory to the image but ignore all files except `.txt` files

### 0.68.15 (2024-12-13)

Adds the `requires_proxy_auth` parameter to `web_endpoint`, `asgi_app`, `wsgi_app`, and `web_server` decorators. Requests to the app will respond with 407 Proxy Authorization Required if a webhook token is not supplied in the HTTP headers. Protects against DoS attacks that will unnecessarily charge users.

### 0.68.11 (2024-12-13)

- `Cls.from_name(...)` now works as a lazy alternative to `Cls.lookup()` that doesn't perform any IO until a method on the class is used for a .remote() call or similar

### 0.68.6 (2024-12-12)

- Fixed a bug introduced in v0.67.47 that suppressed console output from the `modal deploy` CLI.

### 0.68.5 (2024-12-12)

We're removing support for `.spawn()`ing generator functions.

### 0.68.2 (2024-12-11)

- Sandboxes now support a new filesystem API. The `open()` method returns a `FileIO` handle for native file handling in sandboxes.

```python
app = modal.App.lookup("sandbox-fs", create_if_missing=True)
sb = modal.Sandbox.create(app=app)

with sb.open("test.txt", "w") as f:
  f.write("Hello World\n")

f = sb.open("test.txt", "rb")
print(f.read())
```

## 0.67

### 0.67.43 (2024-12-11)

- `modal container exec` and `modal shell` now work correctly even when a pseudoterminal (PTY) is not present. This means, for example, that you can pipe the output of these commands to a file:

  ```python
  modal shell -c 'uv pip list' > env.txt
  ```

### 0.67.39 (2024-12-09)

- It is now possible to delete named `NetworkFileSystem` objects via the CLI (`modal nfs delete ...`) or API `(modal.NetworkFileSystem.delete(...)`)

### 0.67.38 (2024-12-09)

- Sandboxes now support filesystem snapshots. Run `Sandbox.snapshot_filesystem()` to get an Image which can be used to spawn new Sandboxes.

### 0.67.28 (2024-12-05)

- Adds `Image.add_local_python_source` which works similarly to the old and soon-to-be-deprecated `Mount.from_local_python_packages` but for images. One notable difference is that the new `add_local_python_source` _only_ includes `.py`-files by default

### 0.67.23 (2024-12-04)

- Image build functions that use a `functools.wraps` decorator will now have their global variables included in the cache key. Previously, the cache would use global variables referenced within the wrapper itself. This will force a rebuild for Image layers defined using wrapped functions.

### 0.67.22 (2024-12-03)

- Fixed a bug introduced in v0.67.0 where it was impossible to call `modal.Cls` methods when passing a list of requested GPUs.

### 0.67.12 (2024-12-02)

- Fixed a bug that executes the wrong method when a Modal Cls overrides a `@modal.method` inherited from a parent.

### 0.67.7 (2024-11-29)

- Fixed a bug where pointing `modal run` at a method on a Modal Cls would fail if the method was inherited from a parent.

### 0.67.0 (2024-11-27)

New minor client version `0.67.x` comes with an internal data model change for how Modal creates functions for Modal classes. There are no breaking or backwards-incompatible changes with this release. All forward lookup scenarios (`.lookup()` of a `0.67` class from a pre `0.67` client) as well as backwards lookup scenarios (`.lookup()` of a pre `0.67` class from a `0.67` client) work, except for a `0.62` client looking up a `0.67` class (this maintains our current restriction of not being able to lookup a `0.63+` class from a `0.62` client).

## 0.66

### 0.66.49 (2024-11-26)

- `modal config set-environment` will now raise if the requested environment does not exist.

### 0.66.45 (2024-11-26)

- The `modal launch` CLI now accepts a `--detach` flag to run the App in detached mode, such that it will persist after the local client disconnects.

### 0.66.40 (2024-11-23)

- Adds `Image.add_local_file(..., copy=False)` and `Image.add_local_dir(..., copy=False)` as a unified replacement for the old `Image.copy_local_*()` and `Mount.add_local_*` methods.

### 0.66.30 (2024-11-21)

- Removed the `aiostream` package from the modal client library dependencies.

### 0.66.12 (2024-11-19)

`Sandbox.exec` now accepts arguments `text` and `bufsize` for streaming output, which controls text output and line buffering.

### 0.66.0 (2024-11-15)

- Modal no longer supports Python 3.8, which has reached its [official EoL](https://devguide.python.org/versions/).

## 0.65

### 0.65.55 (2024-11-13)

- Escalates stuck input cancellations to container death. This prevents unresponsive user code from holding up resources.
- Input timeouts no longer kill the entire container. Instead, they just cancel the timed-out input, leaving the container and other concurrent inputs running.

### 0.65.49 (2024-11-12)

- Fixed issue in `modal serve` where files used in `Image.copy_*` commands were not watched for changes

### 0.65.42 (2024-11-07)

- `Sandbox.exec` can now accept `timeout`, `workdir`, and `secrets`. See the `Sandbox.create` function for context on how to use these arguments.

### 0.65.33 (2024-11-06)

- Removed the `interactive` parameter from `function` and `cls` decorators. This parameter has been deprecated since May 2024. Instead of specifying Modal Functions as interactive, use `modal run --interactive` to activate interactive mode.

### 0.65.30 (2024-11-05)

- The `checkpointing_enabled` option, deprecated in March 2024, has now been removed.

### 0.65.9 (2024-10-31)

- Output from `Sandbox.exec` can now be directed to `/dev/null`, `stdout`, or stored for consumption. This behavior can be controlled via the new `StreamType` arguments.

### 0.65.8 (2024-10-31)

- Fixed a bug where the `Image.imports` context manager would not correctly propagate ImportError when using a `modal.Cls`.

### 0.65.2 (2024-10-30)

- Fixed an issue where `modal run` would pause for 10s before exiting if there was a failure during app creation.

## 0.64

### 0.64.227 (2024-10-25)

- The `modal container list` CLI command now shows the containers within a specific environment: the active profile's environment if there is one, otherwise the workspace's default environment. You can pass `--env` to list containers in other environments.

### 0.64.223 (2024-10-24)

- Fixed `modal serve` not showing progress when reloading apps on file changes since v0.63.79.

### 0.64.218 (2024-10-23)

- Fix a regression introduced in client version 0.64.209, which affects client authentication within a container.

### 0.64.198 (2024-10-18)

- Fixed a bug where `Queue.put` and `Queue.put_many` would throw `queue.Full` even if `timeout=None`.

### 0.64.194 (2024-10-18)

- The previously-deprecated `--confirm` flag has been removed from the `modal volume delete` CLI. Use `--yes` to force deletion without a confirmation prompt.

### 0.64.193 (2024-10-18)

- Passing `wait_for_response=False` in Modal webhook decorators is no longer supported. See [the docs](https://modal.com/docs/guide/webhook-timeouts#polling-solutions) for alternatives.

### 0.64.187 (2024-10-16)

- When writing to a `StreamWriter` that has already had EOF written, a `ValueError` is now raised instead of an `EOFError`.

### 0.64.185 (2024-10-15)

- Memory snapshotting can now be used with parametrized functions.

### 0.64.184 (2024-10-15)

- StreamWriters now accept strings as input.

### 0.64.182 (2024-10-15)

- Fixed a bug where App rollbacks would not restart a schedule that had been removed in an intervening deployment.

### 0.64.181 (2024-10-14)

- The `modal shell` CLI command now takes a container ID, allowing you to shell into a running container.

### 0.64.180 (2024-10-14)

- `modal shell --cmd` now can be shortened to `modal shell -c`. This means you can use it like `modal shell -c "uname -a"` to quickly run a command within the remote environment.

### 0.64.168 (2024-10-03)

- The `Image.conda`, `Image.conda_install`, and `Image.conda_update_from_environment` methods are now fully deprecated. We recommend using `micromamba` (via `Image.micromamba` and `Image.micromamba_install`) instead, or manually installing and using conda with `Image.run_commands` when strictly necessary.

### 0.64.153 (2024-09-30)

- **Breaking Change:** `Sandbox.tunnels()` now returns a `Dict` rather than a `List`. This dict is keyed by the container's port, and it returns a `Tunnel` object, just like `modal.forward` does.

### 0.64.142 (2024-09-25)

- `modal.Function` and `modal.Cls` now support specifying a `list` of GPU configurations, allowing the Function's container pool to scale across each GPU configuration in preference order.

### 0.64.139 (2024-09-25)

- The deprecated `_experimental_boost` argument is now removed. (Deprecated in late July.)

### 0.64.123 (2024-09-18)

- Sandboxes can now be created without an entrypoint command. If they are created like this, they will stay alive up until their set timeout. This is useful if you want to keep a long-lived sandbox and execute code in it later.

### 0.64.119 (2024-09-17)

- Sandboxes now have a `cidr_allowlist` argument, enabling controlled access to certain IP ranges. When not used (and with `block_network=False`), the sandbox process will have open network access.

### 0.64.118 (2024-09-17)

Introduce an experimental API to allow users to set the input concurrency for a container locally.

### 0.64.112 (2024-09-15)

- Creating sandboxes without an associated `App` is deprecated. If you are spawning a `Sandbox` outside a Modal container, you can lookup an `App` by name to attach to the `Sandbox`:

  ```python
  app = modal.App.lookup('my-app', create_if_missing=True)
  modal.Sandbox.create('echo', 'hi', app=app)
  ```

### 0.64.109 (2024-09-13)

- App handles can now be looked up by name with `modal.App.lookup(name)`. This can be useful for associating Sandboxes with Apps:

  ```python
  app = modal.App.lookup("my-app", create_if_missing=True)
  modal.Sandbox.create("echo", "hi", app=app)
  ```

### 0.64.100 (2024-09-11)

- The default timeout for `modal.Image.run_function` has been lowered to 1 hour. Previously it was 24 hours.

### 0.64.99 (2024-09-11)

- Fixes an issue that could cause containers using `enable_memory_snapshot=True` on Python 3.9 and below to shut down prematurely.

### 0.64.97 (2024-09-11)

- Added support for [ASGI lifespan protocol](https://asgi.readthedocs.io/en/latest/specs/lifespan.html):

  ```python
  @app.function()
  @modal.asgi_app()
  def func():
      from fastapi import FastAPI, Request

      def lifespan(wapp: FastAPI):
          print("Starting")
          yield {"foo": "bar"}
          print("Shutting down")

      web_app = FastAPI(lifespan=lifespan)

      @web_app.get("/")
      def get_state(request: Request):
          return {"message": f"This is the state: {request.state.foo}"}

      return web_app
  ```

  which enables support for `gradio>=v4` amongst other libraries using lifespans

### 0.64.87 (2024-09-05)

- Sandboxes now support port tunneling. Ports can be exposed via the `open_ports` argument, and a list of active tunnels can be retrieved via the `.tunnels()` method.

### 0.64.67 (2024-08-30)

- Fixed a regression in `modal launch` to resume displaying output when starting the container.

### 0.64.48 (2024-08-21)

- Introduces new dataclass-style syntax for class parametrization (see updated [docs](https://modal.com/docs/guide/parametrized-functions))

  ```python
  @app.cls()
  class MyCls:
      param_a: str = modal.parameter()

  MyCls(param_a="hello")  # synthesized constructor
  ```

- The new syntax enforces types (`str` or `int` for now) on all parameters

- _When the new syntax is used_, any web endpoints (`web_endpoint`, `asgi_app`, `wsgi_app` or `web_server`) on the app will now also support parametrization through the use of query parameters matching the parameter names, e.g. `https://myfunc.modal.run/?param_a="hello` in the above example.

- The old explicit `__init__` constructor syntax is still allowed, but could be deprecated in the future and doesn't work with web endpoint parametrization

### 0.64.38 (2024-08-16)

- Added a `modal app rollback` CLI command for rolling back an App deployment to a previous version.

### 0.64.33 (2024-08-16)

- Commands in the `modal app` CLI now accept an App name as a positional argument, in addition to an App ID:

  ```
  modal app history my-app
  ```

  Accordingly, the explicit `--name` option has been deprecated. Providing a name that can be confused with an App ID will also now raise an error.

### 0.64.32 (2024-08-16)

- Updated type stubs using generics to allow static type inferrence for functions calls, e.g. `function.remote(...)`.

### 0.64.26 (2024-08-15)

- `ContainerProcess` handles now support `wait()` and `poll()`, like `Sandbox` objects

### 0.64.24 (2024-08-14)

- Added support for dynamic batching. Functions or class methods decorated with `@modal.batched` will now automatically batch their invocations together, up to a specified `max_batch_size`. The batch will wait for a maximum of `wait_ms` for more invocations after the first invocation is made. See guide for more details.

  ```python
  @app.function()
  @modal.batched(max_batch_size=4, wait_ms=1000)
  async def batched_multiply(xs: list[int], ys: list[int]) -> list[int]:
      return [x * y for x, y in zip(xs, xs)]

  @app.cls()
  class BatchedClass():
      @modal.batched(max_batch_size=4, wait_ms=1000)
      async def batched_multiply(xs: list[int], ys: list[int]) -> list[int]:
          return [x * y for x, y in zip(xs, xs)]
  ```

  The batched function is called with individual inputs:

  ```python
  await batched_multiply.remote.aio(2, 3)
  ```

### 0.64.18 (2024-08-12)

- Sandboxes now have an `exec()` method that lets you execute a command inside the sandbox container. `exec` returns a `ContainerProcess` handle for input and output streaming.

  ```python
  sandbox = modal.Sandbox.create("sleep", "infinity")

  process = sandbox.exec("bash", "-c", "for i in $(seq 1 10); do echo foo $i; sleep 0.5; done")

  for line in process.stdout:
      print(line)
  ```

### 0.64.8 (2024-08-06)

- Removed support for the undocumented `modal.apps.list_apps()` function, which was internal and not intended to be part of public API.

### 0.64.7 (2024-08-05)

- Removed client check for CPU core request being at least 0.1, deferring to server-side enforcement.

### 0.64.2 (2024-08-02)

- Volumes can now be mounted to an ad hoc modal shell session:

  ```
  modal shell --volume my-vol-name
  ```

  When the shell starts, the volume will be mounted at `/mnt/my-vol-name`. This may be helpful for shell-based exploration or manipulation of volume contents.

  Note that the option can be used multiple times to mount additional models:

  ```
  modal shell --volume models --volume data
  ```

### 0.64.0 (2024-07-29)

- App deployment events are now atomic, reducing the risk that a failed deploy will leave the App in a bad state.

## 0.63

### 0.63.87 (2024-07-24)

- The `_experimental_boost` argument can now be removed. Boost is now enabled on all modal Functions.

### 0.63.77 (2024-07-18)

- Setting `_allow_background_volume_commits` is no longer necessary and has been deprecated. Remove this argument in your decorators.

### 0.63.36 (2024-07-05)

- Image layers defined with a `@modal.build` method will now include the values of any _class variables_ that are referenced within the method as part of the layer cache key. That means that the layer will rebuild when the class variables change or are overridden by a subclass.

### 0.63.22 (2024-07-01)

- Fixed an error when running `@modal.build` methods that was introduced in v0.63.19

### 0.63.20 (2024-07-01)

- Fixed bug where `self.method.local()` would re-trigger lifecycle methods in classes

### 0.63.14 (2024-06-28)

- Adds `Cls.lookup()` backwards compatibility with classes created by clients prior to `v0.63`.

  **Important**: When updating (to >=v0.63) an app with a Modal `class` that's accessed using `Cls.lookup()` - make sure to update the client of the app/service **using** `Cls.lookup()` first, and **then** update the app containing the class being looked up.

### 0.63.12 (2024-06-27)

- Fixed a bug introduced in 0.63.0 that broke `modal.Cls.with_options`

### 0.63.10 (2024-06-26)

- Adds warning about future deprecation of `retries` for generators. Retries are being deprecated as they can lead to nondetermistic generator behavior.

### 0.63.9 (2024-06-26)

- Fixed a bug in `Volume.copy_files()` where some source paths may be ignored if passed as `bytes`.
- `Volume.read_file`, `Volume.read_file_into_fileobj`, `Volume.remove_file`, and `Volume.copy_files` can no longer take both string or bytes for their paths. They now only accept `str`.

### 0.63.2 (2024-06-25)

- Fixes issue with `Cls.lookup` not working (at all) after upgrading to v0.63.0. **Note**: this doesn't fix the cross-version lookup incompatibility introduced in 0.63.0.

### 0.63.0 (2024-06-24)

- Changes how containers are associated with methods of `@app.cls()`-decorated Modal "classes".

  Previously each `@method` and web endpoint of a class would get its own set of isolated containers and never run in the same container as other sibling methods.
  Starting in this version, all `@methods` and web endpoints will be part of the same container pool. Notably, this means all methods will scale up/down together, and options like `keep_warm` and `concurrency_limit` will affect the total number of containers for all methods in the class combined, rather than individually.

  **Version incompatibility warning:** Older clients (below 0.63) can't use classes deployed by new clients (0.63 and above), and vice versa. Apps or standalone clients using `Cls.lookup(...)` to invoke Modal classes need to be upgraded to version `0.63` at the same time as the deployed app that's being called into.

- `keep_warm` for classes is now an attribute of the `@app.cls()` decorator rather than individual methods.

## 0.62

### 0.62.236 (2024-06-21)

- Added support for mounting Volume or CloudBucketMount storage in `Image.run_function`. Note that this is _typically_ not necessary, as data downloaded during the Image build can be stored directly in the Image filesystem.

### 0.62.230 (2024-06-18)

- It is now an error to create or lookup Modal objects (`Volume`, `Dict`, `Secret`, etc.) with an invalid name. Object names must be shorter than 64 characters and may contain only alphanumeric characters, dashes, periods, and underscores. The name check had inadvertently been removed for a brief time following an internal refactor and then reintroduced as a warning. It is once more a hard error. Please get in touch if this is blocking access to your data.

### 0.62.224 (2024-06-17)

- The `modal app list` command now reports apps created by `modal app run` or `modal app serve` as being in an "ephemeral" state rather than a "running" state to reduce confusion with deployed apps that are actively processing inputs.

### 0.62.223 (2024-06-14)

- All modal CLI commands now accept `-e` as a short-form of `--env`

### 0.62.220 (2024-06-12)

- Added support for entrypoint and shell for custom containers: `Image.debian_slim().entrypoint([])` can be used interchangeably with `.dockerfile_commands('ENTRYPOINT []')`, and `.shell(["/bin/bash", "-c"])` can be used interchangeably with `.dockerfile_commands('SHELL ["/bin/bash", "-c"]')`

### 0.62.219 (2024-06-12)

- Fix an issue with `@web_server` decorator not working on image builder version 2023.12

### 0.62.208 (2024-06-08)

- `@web_server` endpoints can now return HTTP headers of up to 64 KiB in length. Previously, they were limited to 8 KiB due to an implementation detail.

### 0.62.201 (2024-06-04)

- `modal deploy` now accepts a `--tag` optional parameter that allows you to specify a custom tag for the deployed version, making it easier to identify and manage different deployments of your app.

### 0.62.199 (2024-06-04)

- `web_endpoint`s now have the option to include interactive SwaggerUI/redoc docs by setting `docs=True`
- `web_endpoint`s no longer include an OpenAPI JSON spec route by default

### 0.62.190 (2024-05-29)

- `modal.Function` now supports requesting ephemeral disk (SSD) via the new `ephemeral_disk` parameter. Intended for use in doing large dataset ingestion and transform.

### 0.62.186 (2024-05-29)

- `modal.Volume` background commits are now enabled by default when using `spawn_sandbox`.

### 0.62.185 (2024-05-28)

- The `modal app stop` CLI command now accepts a `--name` (or `-n`) option to stop an App by name rather than by ID.

### 0.62.181 (2024-05-24)

- Background committing on `modal.Volume` mounts is now default behavior.

### 0.62.178 (2024-05-21)

- Added a `modal container stop` CLI command that will kill an active container and reassign its current inputs.

### 0.62.175 (2024-05-17)

- `modal.CloudBucketMount` now supports writing to Google Cloud Storage buckets.

### 0.62.174 (2024-05-17)

- Using `memory=` to specify the type of `modal.gpu.A100` is deprecated in favor of `size=`. Note that `size` accepts a string type (`"40GB"` or `"80GB"`) rather than an integer, as this is a request for a specific variant of the A100 GPU.

### 0.62.173 (2024-05-17)

- Added a `version` flag to the `modal.Volume` API and CLI, allow opting in to a new backend implementation.

### 0.62.172 (2024-05-17)

- Fixed a bug where other functions weren't callable from within an `asgi_app` or `wsgi_app` constructor function and side effects of `@enter` methods weren't available in that scope.

### 0.62.166 (2024-05-14)

- Disabling background commits on `modal.Volume` volumes is now deprecated. Background commits will soon become mandatory behavior.

### 0.62.165 (2024-05-13)

- Deprecated `wait_for_response=False` on web endpoints. See [the docs](https://modal.com/docs/guide/webhook-timeouts#polling-solutions) for alternatives.

### 0.62.162 (2024-05-13)

- A deprecation warning is now raised when using `modal.Stub`, which has been renamed to `modal.App`. Additionally, it is recommended to use `app` as the variable name rather than `stub`, which matters when using the automatic app discovery feature in the `modal run` CLI command.

### 0.62.159 (2024-05-10)

- Added a `--stream-logs` flag to `modal deploy` that, if True, begins streaming the app logs once deployment is complete.

### 0.62.156 (2024-05-09)

- Added support for looking up a deployed App by its deployment name in `modal app logs`

### 0.62.150 (2024-05-08)

- Added validation that App `name`, if provided, is a string.

### 0.62.149 (2024-05-08)

- The `@app.function` decorator now raises an error when it is used to decorate a class (this was always invalid, but previously produced confusing behavior).

### 0.62.148 (2024-05-08)

- The `modal app list` output has been improved in several ways:
  - Persistent storage objects like Volumes or Dicts are no longer included (these objects receive an app ID internally, but this is an implementation detail and subject to future change). You can use the dedicated CLI for each object (e.g. `modal volume list`) instead.
  - For Apps in a _stopped_ state, the output is now limited to those stopped within the past 2 hours.
  - The number of tasks running for each App is now shown.

### 0.62.146 (2024-05-07)

- Added the `region` parameter to the `modal.App.function` and `modal.App.cls` decorators. This feature allows the selection of specific regions for function execution. Note that it is available only on some plan types. See our [blog post](https://modal.com/blog/region-selection-launch) for more details.

### 0.62.144 (2024-05-06)

- Added deprecation warnings when using Python 3.8 locally or in a container. Python 3.8 is nearing EOL, and Modal will be dropping support for it soon.

### 0.62.141 (2024-05-03)

- Deprecated the `Image.conda` constructor and the `Image.conda_install` / `Image.conda_update_from_environment` methods. Conda-based images had a number of tricky issues and were generally slower and heavier than images based on `micromamba`, which offers a similar featureset and can install packages from the same repositories.
- Added the `spec_file` parameter to allow `Image.micromamba_install` to install dependencies from a local file. Note that `micromamba` supports conda yaml syntax along with simple text files.

### 0.62.131 (2024-05-01)

- Added a deprecation warning when object names are invalid. This applies to `Dict`, `NetworkFileSystem`, `Secret`, `Queue`, and `Volume` objects. Names must be shorter than 64 characters and may contain only alphanumeric characters, dashes, periods, and underscores. These rules were previously enforced, but the check had inadvertently been dropped in a recent refactor. Please update the names of your objects and transfer any data to retain access, as invalid names will become an error in a future release.

### 0.62.130 (2024-05-01)

- Added a command-line interface for interacting with `modal.Queue` objects. Run `modal queue --help` in your terminal to see what is available.

### 0.62.116 (2024-04-26)

- Added a command-line interface for interacting with `modal.Dict` objects. Run `modal dict --help` in your terminal to see what is available.

### 0.62.114 (2024-04-25)

- `Secret.from_dotenv` now accepts an optional filename keyword argument:

  ```python
  @app.function(secrets=[modal.Secret.from_dotenv(filename=".env-dev")])
  def run():
      ...
  ```

### 0.62.110 (2024-04-25)

- Passing a glob `**` argument to the `modal volume get` CLI has been deprecated — instead, simply download the desired directory path, or `/` for the entire volume.
- `Volume.listdir()` no longer takes trailing glob arguments. Use `recursive=True` instead.
- `modal volume get` and `modal nfs get` performance is improved when downloading a single file. They also now work with multiple files when outputting to stdout.
- Fixed a visual bug where `modal volume get` on a single file will incorrectly display the destination path.

### 0.62.109 (2024-04-24)

- Improved feedback for deserialization failures when objects are being transferred between local / remote environments.

### 0.62.108 (2024-04-24)

- Added `Dict.delete` and `Queue.delete` as API methods for deleting named storage objects:

  ```python
  import modal
  modal.Queue.delete("my-job-queue")
  ```

- Deprecated invoking `Volume.delete` as an instance method; it should now be invoked as a static method with the name of the Volume to delete, as with the other methods.

### 0.62.98 (2024-04-21)

- The `modal.Dict` object now implements a `keys`/`values`/`items` API. Note that there are a few differences when compared to standard Python dicts:
  - The return value is a simple iterator, whereas Python uses a dictionary view object with more features.
  - The results are unordered.
- Additionally, there was no key data stored for items added to a `modal.Dict` prior to this release, so empty strings will be returned for these entries.

### 0.62.81 (2024-04-18)

- We are introducing `modal.App` as a replacement for `modal.Stub` and encouraging the use of "app" terminology over "stub" to reduce confusion between concepts used in the SDK and the Dashboard. Support for `modal.Stub` will be gradually deprecated over the next few months.

### 0.62.72 (2024-04-16)

- Specifying a hard memory limit for a `modal.Function` is now supported. Pass a tuple of `memory=(request, limit)`. Above the `limit`, which is specified in MiB, a Function's container will be OOM killed.

### 0.62.70 (2024-04-16)

- `modal.CloudBucketMount` now supports read-only access to Google Cloud Storage

### 0.62.69 (2024-04-16)

- Iterators passed to `Function.map()` and similar parallel execution primitives are now executed on the main thread, preventing blocking iterators from possibly locking up background Modal API calls, and risking task shutdowns.

### 0.62.67 (2024-04-15)

- The return type of `Volume.listdir()`, `Volume.iterdir()`, `NetworkFileSystem.listdir()`, and `NetworkFileSystem.iterdir()` is now a `FileEntry` dataclass from the `modal.volume` module. The fields of this data class are the same as the old protobuf object returned by these methods, so it should be mostly backwards-compatible.

### 0.62.65 (2024-04-15)

- Cloudflare R2 bucket support added to `modal.CloudBucketMount`

### 0.62.55 (2024-04-11)

- When Volume reloads fail due to an open file, we now try to identify and report the relevant path. Note that there may be some circumstances in which we are unable to identify the specific file blocking a reload and will report a generic error message in that case.

### 0.62.53 (2024-04-10)

- Values in the `modal.toml` config file that are spelled as `0`, `false`, `"False"`, or `"false"` will now be coerced in Python to`False`, whereas previously only `"0"` (as a string) would have the intended effect.

### 0.62.25 (2024-04-01)

- Fixed a recent regression that caused functions using `modal.interact()` to crash.

### 0.62.15 (2024-03-29)

- Queue methods `put`, `put_many`, `get`, `get_many` and `len` now support an optional `partition` argument (must be specified as a `kwarg`). When specified, users read and write from new partitions of the queue independently. `partition=None` corresponds to the default partition of the queue.

### 0.62.3 (2024-03-27)

- User can now mount S3 buckets using [Requester Pays](https://docs.aws.amazon.com/AmazonS3/latest/userguide/RequesterPaysBuckets.html). This can be done with `CloudBucketMount(..., requester_pays=True)`.

### 0.62.1 (2024-03-27)

- Raise an error on `@web_server(startup_timeout=0)`, which is an invalid configuration.

### 0.62.0 (2024-03-26)

- The `.new()` method has now been deprecated on all Modal objects. It should typically be replaced with `.from_name(...)` in Modal app code, or `.ephemeral()` in scripts that use Modal
- Assignment of Modal objects to a `Stub` via subscription (`stub["object"]`) or attribute (`stub.object`) syntax is now deprecated. This syntax was only necessary when using `.new()`.

## 0.61

### 0.61.104 (2024-03-25)

- Fixed a bug where images based on `micromamba` could fail to build if requesting Python 3.12 when a different version of Python was being used locally.

### 0.61.76 (2024-03-19)

- The `Sandbox`'s `LogsReader` is now an asynchronous iterable. It supports the `async for` statement to stream data from the sandbox's `stdout/stderr`.

```python
@stub.function()
async def my_fn():
    sandbox = stub.spawn_sandbox(
      "bash",
      "-c",
      "while true; do echo foo; sleep 1; done"
    )
    async for message in sandbox.stdout:
        print(f"Message: {message}")
```

### 0.61.57 (2024-03-15)

- Add the `@web_server` decorator, which exposes a server listening on a container port as a web endpoint.

### 0.61.56 (2024-03-15)

- Allow users to write to the `Sandbox`'s `stdin` with `StreamWriter`.

```python
@stub.function()
def my_fn():
    sandbox = stub.spawn_sandbox(
        "bash",
        "-c",
        "while read line; do echo $line; done",
    )
    sandbox.stdin.write(b"foo\\n")
    sandbox.stdin.write(b"bar\\n")
    sandbox.stdin.write_eof()
    sandbox.stdin.drain()
    sandbox.wait()
```

### 0.61.53 (2024-03-15)

- Fixed an bug where` Mount` was failing to include symbolic links.

### 0.61.45 (2024-03-13)

When called from within a container, `modal.experimental.stop_fetching_inputs()` causes it to gracefully exit after the current input has been processed.

### 0.61.35 (2024-03-12)

- The `@wsgi_app()` decorator now uses a different backend based on `a2wsgi` that streams requests in chunks, rather than buffering the entire request body.

### 0.61.32 (2024-03-11)

- Stubs/apps can now be "composed" from several smaller stubs using `stub.include(...)`. This allows more ergonomic setup of multi-file Modal apps.

### 0.61.31 (2024-03-08)

- The `Image.extend` method has been deprecated. This is a low-level interface and can be replaced by other `Image` methods that offer more flexibility, such as `Image.from_dockerfile`, `Image.dockerfile_commands`, or `Image.run_commands`.

### 0.61.24 (2024-03-06)

- Fixes `modal volume put` to support uploading larger files, beyond 40 GiB.

### 0.61.22 (2024-03-05)

- Modal containers now display a warning message if lingering threads are present at container exit, which prevents runner shutdown.

### 0.61.17 (2024-03-05)

- Bug fix: Stopping an app while a container's `@exit()` lifecycle methods are being run no longer interrupts the lifecycle methods.
- Bug fix: Worker preemptions no longer interrupt a container's `@exit()` lifecycle method (until 30 seconds later).
- Bug fix: Async `@exit()` lifecycle methods are no longer skipped for sync functions.
- Bug fix: Stopping a sync function with `allow_concurrent_inputs>1` now actually stops the container. Previously, it would not propagate the signal to worker threads, so they would continue running.
- Bug fix: Input-level cancellation no longer skips the `@exit()` lifecycle method.
- Improve stability of container entrypoint against race conditions in task cancellation.

### 0.61.9 (2024-03-05)

- Fix issue with pdm where all installed packages would be automounted when using package cache (MOD-2485)

### 0.61.6 (2024-03-04)

- For modal functions/classes with `concurrency_limit < keep_warm`, we'll raise an exception now. Previously we (silently) respected the `concurrency_limit` parameter.

### 0.61.1 (2024-03-03)

`modal run --interactive` or `modal run -i` run the app in "interactive mode". This allows any remote code to connect to the user's local terminal by calling `modal.interact()`.

```python
@stub.function()
def my_fn(x):
    modal.interact()

    x = input()
    print(f"Your number is {x}")
```

This means that you can dynamically start an IPython shell if desired for debugging:

```python
@stub.function()
def my_fn(x):
    modal.interact()

    from IPython import embed
    embed()
```

For convenience, breakpoints automatically call `interact()`:

```python
@stub.function()
def my_fn(x):
    breakpoint()
```

## 0.60

### 0.60.0 (2024-02-29)

- `Image.run_function` now allows you to pass args and kwargs to the function. Usage:

```python
def my_build_function(name, size, *, variant=None):
    print(f"Building {name} {size} {variant}")


image = modal.Image.debian_slim().run_function(
    my_build_function, args=("foo", 10), kwargs={"variant": "bar"}
)
```

## 0.59

### 0.59.0 (2024-02-28)

- Mounted packages are now deduplicated across functions in the same stub
- Mounting of local Python packages are now marked as such in the mount creation output, e.g. `PythonPackage:my_package`
- Automatic mounting now includes packages outside of the function file's own directory. Mounted packages are mounted in /root/<module path>

## 0.58

### 0.58.92 (2024-02-27)

- Most errors raised through usage of the CLI will now print a simple error message rather than showing a traceback from inside the `modal` library.
- Tracebacks originating from user code will include fewer frames from within `modal` itself.
- The new `MODAL_TRACEBACK` environment variable (and `traceback` field in the Modal config file) can override these behaviors so that full tracebacks are always shown.

### 0.58.90 (2024-02-27)

- Fixed a bug that could cause `cls`-based functions to to ignore timeout signals.

### 0.58.88 (2024-02-26)

- `volume get` performance is improved for large (> 100MB) files

### 0.58.79 (2024-02-23)

- Support for function parameters in methods decorated with `@exit` has been deprecated. Previously, exit methods were required to accept three arguments containing exception information (akin to `__exit__` in the context manager protocol). However, due to a bug, these arguments were always null. Going forward, `@exit` methods are expected to have no parameters.

### 0.58.75 (2024-02-23)

- Function calls can now be cancelled without killing the container running the inputs. This allows new inputs by different function calls to the same function to be picked up immediately without having to cold-start new containers after cancelling calls.

## 0.57

### 0.57.62 (2024-02-21)

- An `InvalidError` is now raised when a lifecycle decorator (`@build`, `@enter`, or `@exit`) is used in conjunction with `@method`. Previously, this was undefined and could produce confusing failures.

### 0.57.61 (2024-02-21)

- Reduced the amount of context for frames in modal's CLI framework when showing a traceback.

### 0.57.60 (2024-02-21)

- The "dunder method" approach for class lifecycle management (`__build__`, `__enter__`, `__exit__`, etc.) is now deprecated in favor of the modal `@build`, `@enter`, and `@exit` decorators.

### 0.57.52 (2024-02-17)

- In `modal token new` and `modal token set`, the `--no-no-verify` flag has been removed in favor of a `--verify` flag. This remains the default behavior.

### 0.57.51 (2024-02-17)

- Fixes a regression from 0.57.40 where `@enter` methods used a separate event loop.

### 0.57.42 (2024-02-14)

- Adds a new environment variable/config setting, `MODAL_FORCE_BUILD`/`force_build`, that coerces all images to be built from scratch, rather than loaded from cache.

### 0.57.40 (2024-02-13)

- The `@enter()` lifecycle method can now be used to run additional setup code prior to function checkpointing (when the class is decorated with `stub.cls(enable_checkpointing=True)`. Note that there are currently some limitations on function checkpointing:
  - Checkpointing only works for CPU memory; any GPUs attached to the function will not available
  - Networking is disabled while the checkpoint is being created
- Please note that function checkpointing is still a beta feature.

### 0.57.31 (2024-02-12)

- Fixed an issue with displaying deprecation warnings on Windows systems.

### 0.57.22 (2024-02-09)

- Modal client deprecation warnings are now highlighted in the CLI

### 0.57.16 (2024-02-07)

- Fixes a regression in container scheduling. Users on affected versions (**0.57.5**—**0.57.15**) are encouraged to upgrade immediately.

### 0.57.15 (2024-02-07)

- The legacy `image_python_version` config option has been removed. Use the `python_version=` parameter on your image definition instead.

### 0.57.13 (2024-02-07)

- Adds support for mounting an S3 bucket as a volume.

### 0.57.9 (2024-02-07)

- Support for an implicit 'default' profile is now deprecated. If you have more than one profile in your Modal config file, one must be explicitly set to `active` (use `modal profile activate` or edit your `.modal.toml` file to resolve).
- An error is now raised when more than one profile is set to `active`.

### 0.57.2 (2024-02-06)

- Improve error message when generator functions are called with `.map(...)`.

### 0.57.0 (2024-02-06)

- Greatly improved streaming performance of generators and WebSocket web endpoints.
- **Breaking change:** You cannot use `.map()` to call a generator function. (In previous versions, this merged the results onto a single stream, but the behavior was undocumented and not widely used.)
- **Incompatibility:** Generator outputs are now on a different internal system. Modal code on client versions before 0.57 cannot trigger [deployed functions](https://modal.com/docs/guide/trigger-deployed-functions) with `.remote_gen()` that are on client version 0.57, and vice versa.

## 0.56

Note that in version 0.56 and prior, Modal used a different numbering system for patch releases.

### 0.56.4964 (2024-02-05)

- When using `modal token new` or `model token set`, the profile containing the new token will now be activated by default. Use the `--no-activate` switch to update the `modal.toml` file without activating the corresponding profile.

### 0.56.4953 (2024-02-05)

- The `modal profile list` output now indicates when the workspace is determined by a token stored in environment variables.

### 0.56.4952 (2024-02-05)

- Variadic parameters (e.g. \*args and \*\*kwargs) can now be used in scheduled functions as long as the function doesn't have any other parameters without a default value

### 0.56.4903 (2024-02-01)

- `modal container exec`'s `--no-tty` flag has been renamed to `--no-pty`.

### 0.56.4902 (2024-02-01)

- The singular form of the `secret` parameter in `Stub.function`, `Stub.cls`, and `Image.run_function` has been deprecated. Please update your code to use the plural form instead:`secrets=[Secret(...)]`.

### 0.56.4885 (2024-02-01)

- In `modal profile list`, the user's GitHub username is now shown as the name for the "Personal" workspace.

### 0.56.4874 (2024-01-31)

- The `modal token new` and `modal token set` commands now create profiles that are more closely associated with workspaces, and they have more explicit profile activation behavior:
  - By default, these commands will create/update a profile named after the workspace that the token points to, rather than a profile named "default"
  - Both commands now have an `--activate` flag that will activate the profile associated with the new token
  - If no other profiles exist at the time of creation, the new profile will have its `active` metadata set to True
- With these changes, we are moving away from the concept of a "default" profile. Implicit usage of the "default" profile will be deprecated in a future update.

### 0.56.4849 (2024-01-29)

- Adds tty support to `modal container exec` for fully-interactive commands. Example: `modal container exec [container-id] /bin/bash`

### 0.56.4792 (2024-01-26)

- The `modal profile list` command now shows the workspace associated with each profile.

### 0.56.4715 (2024-01-24)

- `Mount.from_local_python_packages` now places mounted packages at `/root` in the Modal runtime by default (used to be `/pkg`). To override this behavior, the function now takes a `remote_dir: Union[str, PurePosixPath]` argument.

### 0.56.4707 (2024-01-23)

- The Modal client library is now compatible with Python 3.12, although there are a few limitations:

  - Images that use Python 3.12 without explicitly specifing it through `python_version` or `add_python` will not build
    properly unless the modal client is also running on Python 3.12.
  - The `conda` and `microconda` base images currently do not support Python 3.12 because an upstream dependency is not yet compatible.

### 0.56.4700 (2024-01-22)

- `gpu.A100` class now supports specifying GiB memory configuration using a `size: str` parameter. The `memory: int` parameter is deprecated.

### 0.56.4693 (2024-01-22)

- You can now execute commands in running containers with `modal container exec [container-id] [command]`.

### 0.56.4691 (2024-01-22)

- The `modal` cli now works more like the `python` cli in regard to script/module loading:
  - Running `modal my_dir/my_script.py` now puts `my_dir` on the PYTHONPATH.
  - `modal my_package.my_module` will now mount to /root/my_package/my_module.py in your Modal container, regardless if using automounting or not (and any intermediary `__init__.py` files will also be mounted)

### 0.56.4687 (2024-01-20)

- Modal now uses the current profile if `MODAL_PROFILE` is set to the empty string.

### 0.56.4649 (2024-01-17)

- Dropped support for building Python 3.7 based `modal.Image`s. Python 3.7 is end-of-life since late June 2023.

### 0.56.4620 (2024-01-16)

- modal.Stub.function now takes a `block_network` argument.

### 0.56.4616 (2024-01-16)

- modal.Stub now takes a `volumes` argument for setting the default volumes of all the stub's functions, similarly to the `mounts` and `secrets` argument.

### 0.56.4590 (2024-01-13)

- `modal serve`: Setting MODAL_LOGLEVEL=DEBUG now displays which files cause an app reload during serve

### 0.56.4570 (2024-01-12)

- `modal run` cli command now properly propagates `--env` values to object lookups in global scope of user code
