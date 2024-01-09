# Changelog

The Modal client library is still in pre-1.0 development. Patch releases are made on every change.

We will update this changelog as we release new _minor_ versions.

Sometimes breaking changes are necessary, but we will try to minimize them and provide migration guides and deprecation warnings several versions in advance of removing functionality. This gradually takes place over several months.

We appreciate your patience while we speedily work towards a stable release of the client.

**Historical support matrix:**

| Date              | Supported version | Recommended version | Latest version |
| ----------------- | ----------------: | ------------------: | -------------: |
| January 4, 2024   |         0.51.3328 |           0.55.3849 |      0.56.4434 |
| November 24, 2023 |         0.49.2343 |           0.54.3748 |      0.55.4117 |

## Latest (will be marked as v0.57.X)

There are no breaking changes in this version.

We added support for a [new lifecycle method API](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-functions-and-parameters) based on class decorators. Users are advised to move to the new API rather than the previous one, based on the `__enter__()` and `__exit__()` hooks.

```python
from modal import Stub, build, enter, exit

@stub.cls()
class MyLifecycle:
    @build()
    @enter()
    def runs_on_build_and_startup():
        self.foo = load_expensive_model()

    @exit()
    def runs_on_shutdown():
        self.foo.destroy()
        track_finish()
```

We deprecated `gpu=True` on functions. Use `gpu="any"` or `gpu=modal.gpu.Any()` instead.

```python
@stub.function(gpu="any")
def my_function():
    pass
```

We deprecated the `stub.is_inside()` API. Use lifecycle methods or `with image.imports()` instead.

```python
with image.imports():
    import my_module
```

We deprecated the experimental `inf2` GPU type for Inferentia 2 chips. This machine type was under a closed beta.

- **New features:**
  - Client support for WebSockets in web endpoints (#1110)
  - Add size to VolumeGet (#1117)
  - [MOD 1731] vol get progress display (#1118)
  - Exposes memory checkpointing flag (#1120)
  - [Feature] add blocking modal.Queue.put to handle full queues (#1132)
  - Returns default value when using Modal.Dict.get() (#1139)
- **Bug fixes:**
  - Improve pip_install_from_pyproject err handling (#1111)
  - improve exceptions in .read_file methods (#1113)
  - Re-raise exception on Python 3.12 (#1114)
  - Cancel an ASGI app if no input is received in 5 seconds. (#1125)
- **Improvements:**
  - Increase HTTP/2 gRPC window size to 64 MiB (#1107)
  - perf: intro --compile option to poetry installs (#1112)
  - Add APP_STATE_DETACHED_DISCONNECTED (#1115)
  - [MOD-1995] better err message for missing vol (#1121)
  - Track and hydrate object dependencies of functions using closurevars (#1116)
  - Expand width of DataChunk index to 64-bit (#1123)
  - Track objects based on id (#1124)
  - Adds AppState to checkpoint info (#1127)

## v0.56.4220 (Dec 6, 2023)

Initial minor release tracked in changelog.
