# Changelog

This changelog documents user-facing updates (features, enhancements, fixes, and deprecations) to the `modal` client library. Patch releases are made on every change.

The client library is still in pre-1.0 development, and sometimes breaking changes are necessary. We try to minimize them and publish deprecation warnings / migration guides in advance, typically providing a transition window of several months.

You can also refer to the [1.0 migration guide](https://modal.com/docs/guide/modal-1-0-migration) for a summary of major breaking changes that we are rolling out before releasing version 1.0.

We appreciate your patience while we speedily work towards a stable release of the client.

## Latest

<!-- NEW CONTENT GENERATED BELOW. PLEASE PRESERVE THIS COMMENT. -->

### 0.73.84 (2025-03-04)

- The `keep_warm=`  parameter has been removed from the`@modal.method` decorator. This parameter has been nonfunctional since v0.63.0; all autoscaler configuration must be done at the level of the modal Cls.



### 0.73.82 (2025-03-04)

- Adds `modal.fastapi_endpoint` as an alias for `modal.web_endpoint`. We will be deprecating the `modal.web_endpoint`  _name_ (but not the functionality) as part of the Modal 1.0 release.



### 0.73.81 (2025-03-03)

- The `wait_for_response` parameter of Modal's web endpoint decorators has been removed (originally deprecated in May 2024).



### 0.73.78 (2025-03-01)

- It is now possible to call `Cls.with_options` on an unhydrated Cls, e.g.

    ```python
    ModelWithGPU = modal.Cls.from_name("my-app", "Model").with_options(gpu="H100")
    ```



### 0.73.77 (2025-03-01)

* `Cls.with_options()` now accept unhydated volume and secrets



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

* Fixes issue where running `modal shell` with a dot-separated module reference as input would not accept the required `-m` flag for "module mode", but still emitted a warning telling users to use `-m`



### 0.73.60 (2025-02-20)

* Fixes an issue where `modal.runner.deploy_app()` didn't work when called from within a running (remote) Modal Function



### 0.73.58 (2025-02-20)

* Introduces an `-m` flag to `modal run`, `modal shell`, `modal serve` and `modal deploy`, which indicates that the modal app/function file is specified using python "module syntax" rather than a file path. In the future this will be a required flag when using module syntax.

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

* Deprecate the GPU classes (`gpu=A100(...)` etc) in favor of just using strings (`gpu="A100"` etc)



### 0.73.26 (2025-02-10)

* Adds a pending deprecation warning when looking up class methods using `Function.from_name`, e.g. `Function.from_name("some_app", "SomeClass.some_method")`. The recommended way to reference methods of classes is to look up the class instead: `RemoteClass = Cls.from_name("some_app", "SomeClass")`



### 0.73.25 (2025-02-09)

- Fixes an issue introduced in `0.73.19` that prevented access to GPUs during image builds



### 0.73.18 (2025-02-06)

* When using a parameterized class (with at least one `modal.parameter()` specified), class instantiation with an incorrect construction signature (wrong arguments or types) will now fail at the `.remote()` calling site instead of container startup for the called class.



### 0.73.14 (2025-02-04)

- Fixed the status message shown in terminal logs for ephemeral Apps to accurately report the number of active containers.



### 0.73.11 (2025-02-04)

* Warns users if the `modal.Image` of a Function/Cls doesn't include all the globally imported "local" modules (using `.add_local_python_source()`), and the user hasn't explicitly set an `include_source` value of True/False. This is in preparation for an upcoming deprecation of the current "auto mount" logic.



### 0.73.10 (2025-02-04)

* Modal functions, methods and entrypoints can now accept variable-length arguments to skip Modal's default CLI parsing. This is useful if you want to use Modal with custom argument parsing via `argparse` or `HfArgumentParser`. For example, the following function can be invoked with `modal run my_file.py --foo=42 --bar="baz"`:

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

* `modal run` now runs a single local entrypoints/function in the selected module. If exactly one local entrypoint or function exists in the selected module, the user doesn't have to qualify the runnable
in the modal run command, even if some of the module's referenced apps have additional local entrypoints or functions. This partially restores "auto-inferred function" functionality that was changed in v0.72.48.



### 0.73.0 (2025-01-30)

* Introduces an `include_source` argument in the `App.function` and `App.cls` decorators that let users configure which class of python packages are automatically included as source mounts in created modal functions/classes (what we used to call "automount" behavior). This will supersede the MODAL_AUTOMOUNT configuration value which will eventually be deprecated. As a convenience, the `modal.App` constructor will also accept an `include_source` argument which serves as the default for all the app's functions and classes.
  
  The `include_source` argument accepts the following values:
  * `True` (default in a future version of Modal) Automatically includes the Python files of the source package of the function's own home module, but not any other local packages. Roughly equivalent ot `MODAL_AUTOMOUNT=0` in previous versions of Modal.
  * `False` - don't include *any* local source. Assumes the function's home module is importable in the container environment through some other means (typically added to the provided `modal.Image`'s Python environment).
  * `None` (the default) - use current soon-to-be-deprecated automounting behavior, including source of all first party packages that are not installed into site-packages locally.
  
* Minor change to `MODAL_AUTOMOUNT=0`:  When running/deploying using a module path (e.g. `modal run mypak.mymod`), **all non .pyc files** of the source package (`mypak` in this case) are now included in the function's container. Previously, only the function's home `.py` module file + any `__init__.py` files in its package structure were included. Note that this is only for MODAL_AUTOMOUNT=0. To get full control over which source files are included with your functions, you can set `include_source=False` on your function (see above) and manually specify the files to include using the `ignore` argument to `Image.add_local_python_source`.


## 0.72


### 0.72.56 (2025-01-28)

- Deprecated `.lookup` methods on Modal objects. Users are encouraged to use `.from_name` instead. In most cases this will be a simple name substitution. See [the 1.0 migration guide](https://modal.com/docs/guide/modal-1-0-migration#deprecating-the-lookup-method-on-modal-objects) for more information.



### 0.72.54 (2025-01-28)

* Fixes bug introduced in v0.72.48 where `modal run` didn't work with files having global `Function.from_name()`/`Function.lookup()`/`Cls.from_name()`/`Cls.lookup()` calls.



### 0.72.48 (2025-01-24)

* Fixes a CLI bug where you couldn't reference functions via a qualified app, e.g. `mymodule::{app_variable}.{function_name}`.
* The `modal run`, `modal serve` and `modal shell` commands get more consistent error messages in cases where the passed app or function reference isn't resolvable to something that the current command expects.
* Removes the deprecated `__getattr__`, `__setattr__`, `__getitem__` and `__setitem__` methods from `modal.App`



### 0.72.39 (2025-01-22)

- Introduced a new public method, `.hydrate`, for on-demand hydration of Modal objects. This method replaces the existing semi-public `.resolve` method, which is now deprecated.



### 0.72.33 (2025-01-20)

* The Image returned by `Sandbox.snapshot_filesystem` now has `object_id` and other metadata pre-assigned rather than require loading by subsequent calls to sandboxes or similar to set this data.



### 0.72.30 (2025-01-18)

* Adds a new `oidc_auth_role_arn` field to `CloudBucketMount` for using OIDC authentication to create the mountpoint.



### 0.72.24 (2025-01-17)

* No longer prints a warning if `app.include` re-includes an already included function (warning is still printed if *another* function with the same name is included)



### 0.72.22 (2025-01-17)

* Internal refactor of the `modal.object` module. All entities except `Object` from that module have now been moved to the `modal._object` "private" module.



### 0.72.17 (2025-01-16)

- The `@modal.build` decorator is now deprecated. For storing large assets (e.g. model weights), we now recommend using a `modal.Volume` over writing data to the `modal.Image` filesystem directly.



### 0.72.16 (2025-01-16)

* Fixes bug introduced in v0.72.9 where `modal run SomeClass.some_method` would incorrectly print a deprecation warning.



### 0.72.15 (2025-01-15)

- Added an `environment_name` parameter to the `App.run` context manager.



### 0.72.8 (2025-01-10)

- Fixes a bug introduced in v0.72.2 when specifying `add_python="3.9"` in `Image.from_registry`.



### 0.72.0 (2025-01-09)

* The default behavior`Image.from_dockerfile()` and `image.dockerfile_commands()` if no parameter is passed to `ignore` will be to automatically detect if there is a valid dockerignore file in the current working directory or next to the dockerfile following the same rules as `dockerignore` does using `docker` commands. Previously no patterns were ignored.


## 0.71


### 0.71.13 (2025-01-09)

* `FilePatternMatcher` has a new constructor `from_file` which allows you to read file matching patterns from a file instead of having to pass them in directly, this can be used for `Image` methods accepting an `ignore` parameter in order to read ignore patterns from files.



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

* `Image.from_dockerfile()` and `image.dockerfile_commands()` now auto-infer which files need to be uploaded based on COPY commands in the source if `context_mount` is omitted. The `ignore=` argument to these methods can be used to selectively omit files using a set of glob patterns.


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

* `Cls.from_name(...)` now works as a lazy alternative to `Cls.lookup()` that doesn't perform any IO until a method on the class is used for a .remote() call or similar



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

* Adds `Image.add_local_python_source` which works similarly to the old and soon-to-be-deprecated `Mount.from_local_python_packages` but for images. One notable difference is that the new `add_local_python_source` *only* includes `.py`-files by default



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

* Adds `Image.add_local_file(..., copy=False)` and `Image.add_local_dir(..., copy=False)` as a unified replacement for the old `Image.copy_local_*()` and `Mount.add_local_*` methods.



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

* Fixed issue in `modal serve` where files used in `Image.copy_*` commands were not watched for changes



### 0.65.42 (2024-11-07)

- `Sandbox.exec` can now accept `timeout`, `workdir`, and `secrets`. See the `Sandbox.create` function for context on how to use these arguments.



### 0.65.33 (2024-11-06)

- Removed the `interactive` parameter from `function` and `cls` decorators. This parameter has been deprecated since May 2024. Instead of specifying Modal Functions as interactive, use `modal run --interactive` to activate interactive mode.



### 0.65.30 (2024-11-05)

* The `checkpointing_enabled` option, deprecated in March 2024, has now been removed.



### 0.65.9 (2024-10-31)

- Output from `Sandbox.exec` can now be directed to `/dev/null`, `stdout`, or stored for consumption. This behavior can be controlled via the new `StreamType` arguments.



### 0.65.8 (2024-10-31)

- Fixed a bug where the `Image.imports` context manager would not correctly propagate ImportError when using a `modal.Cls`.



### 0.65.2 (2024-10-30)

* Fixed an issue where `modal run` would pause for 10s before exiting if there was a failure during app creation.


## 0.64


### 0.64.227 (2024-10-25)

- The `modal container list` CLI command now shows the containers within a specific environment: the active profile's environment if there is one, otherwise the workspace's default environment. You can pass `--env` to list containers in other environments.


### 0.64.223 (2024-10-24)

* Fixed `modal serve` not showing progress when reloading apps on file changes since v0.63.79.



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

* `modal.Function` and `modal.Cls` now support specifying a `list` of GPU configurations, allowing the Function's container pool to scale across each GPU configuration in preference order.



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

* The default timeout for `modal.Image.run_function` has been lowered to 1 hour. Previously it was 24 hours.



### 0.64.99 (2024-09-11)

* Fixes an issue that could cause containers using `enable_memory_snapshot=True` on Python 3.9 and below to shut down prematurely.


### 0.64.97 (2024-09-11)

* Added support for [ASGI lifespan protocol](https://asgi.readthedocs.io/en/latest/specs/lifespan.html): 

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

- *When the new syntax is used*, any web endpoints (`web_endpoint`, `asgi_app`, `wsgi_app` or `web_server`) on the app will now also support parametrization through the use of query parameters matching the parameter names, e.g. `https://myfunc.modal.run/?param_a="hello` in the above example.

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

- Added support for dynamic batching. Functions or class methods decorated with `@modal.batched` will now automatically batch their invocations together, up to a specified `max_batch_size`.  The batch will wait for a maximum of `wait_ms` for more invocations after the first invocation is made. See guide for more details.

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

* The `_experimental_boost` argument can now be removed. Boost is now enabled on all modal Functions.



### 0.63.77 (2024-07-18)

* Setting `_allow_background_volume_commits` is no longer necessary and has been deprecated. Remove this argument in your decorators.



### 0.63.36 (2024-07-05)

- Image layers defined with a `@modal.build` method will now include the values of any _class variables_ that are referenced within the method as part of the layer cache key. That means that the layer will rebuild when the class variables change or are overridden by a subclass.


### 0.63.22 (2024-07-01)

* Fixed an error when running `@modal.build` methods that was introduced in v0.63.19



### 0.63.20 (2024-07-01)

* Fixed bug where `self.method.local()` would re-trigger lifecycle methods in classes



### 0.63.14 (2024-06-28)

* Adds `Cls.lookup()` backwards compatibility with classes created by clients prior to `v0.63`.

    **Important**: When updating (to >=v0.63) an app with a Modal `class` that's accessed using `Cls.lookup()` - make sure to update the client of the app/service **using** `Cls.lookup()` first, and **then** update the app containing the class being looked up.



### 0.63.12 (2024-06-27)

- Fixed a bug introduced in 0.63.0 that broke `modal.Cls.with_options`



### 0.63.10 (2024-06-26)

- Adds warning about future deprecation of `retries` for generators. Retries are being deprecated as they can lead to nondetermistic generator behavior.



### 0.63.9 (2024-06-26)

- Fixed a bug in `Volume.copy_files()` where some source paths may be ignored if passed as `bytes`.
- `Volume.read_file`, `Volume.read_file_into_fileobj`, `Volume.remove_file`, and `Volume.copy_files` can no longer take both string or bytes for their paths. They now only accept `str`.



### 0.63.2 (2024-06-25)

* Fixes issue with `Cls.lookup` not working (at all) after upgrading to v0.63.0. **Note**: this doesn't fix the cross-version lookup incompatibility introduced in 0.63.0.



### 0.63.0 (2024-06-24)

* Changes how containers are associated with methods of `@app.cls()`-decorated Modal "classes".

    Previously each `@method` and web endpoint of a class would get its own set of isolated containers and never run in the same container as other sibling methods.
    Starting in this version, all `@methods` and web endpoints will be part of the same container pool. Notably, this means all methods will scale up/down together, and options like `keep_warm` and `concurrency_limit` will affect the total number of containers for all methods in the class combined, rather than individually.

    **Version incompatibility warning:** Older clients (below 0.63) can't use classes deployed by new clients (0.63 and above), and vice versa. Apps or standalone clients using `Cls.lookup(...)` to invoke Modal classes need to be upgraded to version `0.63` at the same time as the deployed app that's being called into.

* `keep_warm` for classes is now an attribute of the `@app.cls()` decorator rather than individual methods.


## 0.62


### 0.62.236 (2024-06-21)

- Added support for mounting Volume or CloudBucketMount storage in `Image.run_function`. Note that this is *typically* not necessary, as data downloaded during the Image build can be stored directly in the Image filesystem.



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

* `modal.Function` now supports requesting ephemeral disk (SSD) via the new `ephemeral_disk` parameter. Intended for use in doing large dataset ingestion and transform.



### 0.62.186 (2024-05-29)

* `modal.Volume` background commits are now enabled by default when using `spawn_sandbox`.



### 0.62.185 (2024-05-28)

- The `modal app stop` CLI command now accepts a `--name` (or `-n`) option to stop an App by name rather than by ID.



### 0.62.181 (2024-05-24)

* Background committing on `modal.Volume` mounts is now default behavior.



### 0.62.178 (2024-05-21)

- Added a `modal container stop` CLI command that will kill an active container and reassign its current inputs.



### 0.62.175 (2024-05-17)

* `modal.CloudBucketMount` now supports writing to Google Cloud Storage buckets.



### 0.62.174 (2024-05-17)

- Using `memory=` to specify the type of `modal.gpu.A100` is deprecated in favor of `size=`. Note that `size` accepts a string type (`"40GB"` or `"80GB"`) rather than an integer, as this is a request for a specific variant of the A100 GPU.



### 0.62.173 (2024-05-17)

- Added a `version` flag to the `modal.Volume` API and CLI, allow opting in to a new backend implementation.



### 0.62.172 (2024-05-17)

* Fixed a bug where other functions weren't callable from within an `asgi_app` or `wsgi_app` constructor function and side effects of `@enter` methods weren't available in that scope.



### 0.62.166 (2024-05-14)

* Disabling background commits on `modal.Volume` volumes is now deprecated. Background commits will soon become mandatory behavior.



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
    - For Apps in a *stopped* state, the output is now limited to those stopped within the past 2 hours.
    - The number of tasks running for each App is now shown.



### 0.62.146 (2024-05-07)

- Added the `region` parameter to the `modal.App.function`  and `modal.App.cls` decorators. This feature allows the selection of specific regions for function execution. Note that it is available only on some plan types. See our [blog post](https://modal.com/blog/region-selection-launch) for more details.




### 0.62.144 (2024-05-06)

* Added deprecation warnings when using Python 3.8 locally or in a container. Python 3.8 is nearing EOL, and Modal will be dropping support for it soon.



### 0.62.141 (2024-05-03)

* Deprecated the `Image.conda` constructor and the `Image.conda_install` / `Image.conda_update_from_environment` methods. Conda-based images had a number of tricky issues and were generally slower and heavier than images based on `micromamba`, which offers a similar featureset and can install packages from the same repositories.
* Added the `spec_file` parameter to allow `Image.micromamba_install` to install dependencies from a local file. Note that `micromamba` supports conda yaml syntax along with simple text files.



### 0.62.131 (2024-05-01)

* Added a deprecation warning when object names are invalid. This applies to `Dict`, `NetworkFileSystem`, `Secret`, `Queue`, and `Volume` objects. Names must be shorter than 64 characters and may contain only alphanumeric characters, dashes, periods, and underscores. These rules were previously enforced, but the check had inadvertently been dropped in a recent refactor.  Please update the names of your objects and transfer any data to retain access, as invalid names will become an error in a future release.



### 0.62.130 (2024-05-01)

- Added a command-line interface for interacting with `modal.Queue` objects. Run `modal queue --help` in your terminal to see what is available.



### 0.62.116 (2024-04-26)

* Added a command-line interface for interacting with `modal.Dict` objects. Run `modal dict --help` in your terminal to see what is available.



### 0.62.114 (2024-04-25)

* `Secret.from_dotenv` now accepts an optional filename keyword argument:

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
- Deprecated invoking `Volume.delete` as an instance method; it should now be invoked as a static method with the name of  the Volume to delete, as with the other methods.



### 0.62.98 (2024-04-21)

- The `modal.Dict` object now implements a `keys`/`values`/`items` API. Note that there are a few differences when compared to standard Python dicts:
    - The return value is a simple iterator, whereas Python uses a dictionary view object with more features.
    - The results are unordered.
 - Additionally, there was no key data stored for items added to a `modal.Dict` prior to this release, so empty strings will be returned for these entries.



### 0.62.81 (2024-04-18)

* We are introducing `modal.App` as a replacement for `modal.Stub` and encouraging the use of "app" terminology over "stub" to reduce confusion between concepts used in the SDK and the Dashboard. Support for `modal.Stub` will be gradually deprecated over the next few months.



### 0.62.72 (2024-04-16)

* Specifying a hard memory limit for a `modal.Function` is now supported. Pass a tuple of `memory=(request, limit)`. Above the `limit`, which is specified in MiB, a Function's container will be OOM killed.



### 0.62.70 (2024-04-16)

* `modal.CloudBucketMount` now supports read-only access to Google Cloud Storage



### 0.62.69 (2024-04-16)

* Iterators passed to `Function.map()` and similar parallel execution primitives are now executed on the main thread, preventing blocking iterators from possibly locking up background Modal API calls, and risking task shutdowns.



### 0.62.67 (2024-04-15)

- The return type of `Volume.listdir()`, `Volume.iterdir()`, `NetworkFileSystem.listdir()`, and `NetworkFileSystem.iterdir()` is now a `FileEntry` dataclass from the `modal.volume` module. The fields of this data class are the same as the old protobuf object returned by these methods, so it should be mostly backwards-compatible.



### 0.62.65 (2024-04-15)

* Cloudflare R2 bucket support added to `modal.CloudBucketMount`



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

* Stubs/apps can now be "composed" from several smaller stubs using `stub.include(...)`. This allows more ergonomic setup of multi-file Modal apps.



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

* Fix issue with pdm where all installed packages would be automounted when using package cache (MOD-2485)



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

* Mounted packages are now deduplicated across functions in the same stub
* Mounting of local Python packages are now marked as such in the mount creation output, e.g. `PythonPackage:my_package`
* Automatic mounting now includes packages outside of the function file's own directory. Mounted packages are mounted in /root/<module path>



## 0.58


### 0.58.92 (2024-02-27)

- Most errors raised through usage of the CLI will now print a simple error message rather than showing a traceback from inside the `modal` library.
- Tracebacks originating from user code will include fewer frames from within `modal` itself.
- The new `MODAL_TRACEBACK` environment variable (and `traceback` field in the Modal config file) can override these behaviors so that full tracebacks are always shown.


### 0.58.90 (2024-02-27)

- Fixed a bug that could cause `cls`-based functions to to ignore timeout signals.



### 0.58.88 (2024-02-26)

* `volume get` performance is improved for large (> 100MB) files



### 0.58.79 (2024-02-23)

* Support for function parameters in methods decorated with `@exit` has been deprecated. Previously, exit methods were required to accept three arguments containing exception information (akin to `__exit__` in the context manager protocol). However, due to a bug, these arguments were always null. Going forward, `@exit` methods are expected to have no parameters.



### 0.58.75 (2024-02-23)

* Function calls can now be cancelled without killing the container running the inputs. This allows new inputs by different function calls to the same function to be picked up immediately without having to cold-start new containers after cancelling calls.



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

- Variadic parameters (e.g. *args and **kwargs) can now be used in scheduled functions as long as the function doesn't have any other parameters without a default value



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


