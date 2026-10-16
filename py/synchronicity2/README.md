# Synchronicity2

Synchronicity2 is a utility that generates consistent and type safe synchronous and asynchronous interfaces from a single async implementation.

## Intro
### TLDR

In short, an async API implementation like:

```py notest
async def foo_impl() -> str:
    return await external_resource()
```

Becomes a public interface that is usable like this:
```py notest
from public_api import foo

foo()  # calls foo_impl above without async syntax

await foo.aio()  # but you can also call it async!
```

In addition to simple coroutine functions like above, synchronicity2 supports wrapping classes with async methods, async iterators, two-way async generators, async context managers and more.

### Usage pattern

At a high level, you use it like this:

- Write implementation code as normal async Python (`async def foo()`) with no special instrumentation apart from regular Python type annotations
- Instrument your code with information about where you want your public interface to this async code
- Generate the public sync + async API of your package using the `synchronicity2` CLI as part of your build or packaging step.
- Every function or method now gets two type safe call paths:
  - a blocking sync interface like `mylib.foo(...)`
  - an async interface like `await mylib.foo.aio(...)`

## Usage Guide
### Install the synchronicity2 package as a dev dependency

```bash
uv add --dev synchronicity2
```

(or just `pip install synchronicity2` - the point is just to get the package into your dev python env)

### Instrument your package

Enabling synchronicity2 exports requires only very light weight instrumentation of your code: adding `synchronicity2.Module("output_module_name")` declarations that tell the codegen tools which output modules to put wrappers into.

You then attach functions and types to these Modules through use of decorators. 

Here is a simple example of how this looks in practice for a dummy weather module (note that function bodies have no impact)

```python
# mylib/_weather_impl.py
from mylib.synchronicity2 import Module

wrapper_module = Module("mylib.weather")


@wrapper_module.wrap_function()
async def get_temperature(city: str) -> float: ...


@wrapper_module.wrap_class()
class WeatherClient:
    default_city: str

    def __init__(self, default_city: str):
        self.default_city = default_city

    async def current(self) -> float: ...
```

In addition to the `Module` declarations, synchronicity2 relies on correct type annotations of function arguments and return values to determine what wrapper logic needs to do. This is particularly important for types that directly or indirectly indicate async functionality in the types (e.g. async generators or other types that themselves have async methods):

```python continuation
import collections.abc


@wrapper_module.wrap_function()
async def stream_temperature_readings() -> collections.abc.AsyncGenerator[float, None]:
    """Async generator of sample readings"""
    yield 17.5
    yield 18.0
    yield 18.5
```

`Module` registration is build-time metadata only: the decorators return the original function or class unchanged, without starting a synchronizer or changing how your implementation code runs.

**Recommendations when authoring implementation code**

- Add type annotations. Generation relies heavily on them.
- Don't make implementation modules depend on wrapper types, especially not in global scope (otherwise your implementation isn't importable during code gen before codegen has run)
- Run unit tests on implementation code instead of wrapper code - it's nice to avoid having to run codegen between iterations
- You can't delete implementation modules after generating wrappers - the wrappers import them to delegate execution there

### Generate wrappers
Point the `synchronicity2 wrappers` codegen utility to your instrumented modules to generate the declared output modules (as specified by your `Module` declarations)

```bash
synchronicity2 wrappers \
  --synchronizer-module mylib._synchronizer \
  --runtime-package mylib.synchronicity2 \
  -m mylib._weather_impl \
  -o src
```

The above example creates two files:
* `src/mylib/weather.py` - this contains the wrappers. If you have multiple Module declarations in your code you can many such files by a single invocation. You also can (and should) point to multiple instrumented modules with many `-m` in the same cli invocation.
* `src/mylib/_synchronizer.py` - this is the singleton "synchronizer module" as declared in the CLI above and is used by all generated wrapper modules.

## Advanced usage

### Including unwrapped entries

If a generated module should expose a plain module-level value without generating a synchronicity2 wrapper for it, use `Module.manual_export(...)`. This is intended for constants, type aliases, helper dataclasses, enums, and helper functions that should remain importable from the generated public module but do not need sync/async or wrapper/implementation translation.

```python
from dataclasses import dataclass

from mylib.synchronicity2 import Module

mod = Module("mylib.api")

DEFAULT_TIMEOUT = 30


@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int


def validate_name(name: str) -> None:
    if not name:
        raise ValueError("name must not be empty")


mod.manual_export("DEFAULT_TIMEOUT")
mod.manual_export("FileInfo")
mod.manual_export("validate_name")
```

The generated module emits direct aliases such as `DEFAULT_TIMEOUT = mylib._impl.DEFAULT_TIMEOUT`. `manual_export(...)` does not parse annotations, does not create a wrapper class or wrapper function, and does not add call-boundary translation.

Use `source_name=` when the public export name differs from the implementation name, or `source_module=` when forwarding a name from another module:

```python
from mylib.synchronicity2 import Module

mod = Module("mylib.api")

mod.manual_export("public_name", source_name="_private_name")
mod.manual_export("OTHER_VALUE", source_module="mylib._shared")
```

If a generated module or wrapper class should expose a wrapper-aware entry without generating a normal synchronicity2 wrapper body, register it as usual and mark the inserted object with `Module.manual_wrapper()`.

```python
from mylib.synchronicity2 import Module
from mylib.synchronicity2.descriptor import FunctionWithAio, MethodWithAio, function_with_aio, method_with_aio

mod = Module("mylib.api")


class _ManualFunctionWithAio(FunctionWithAio):
    def __call__(self, value: int) -> str:
        return self._sync_impl(value)

    async def aio(self, value: int) -> str:
        return f"aio:{value}"


@mod.wrap_function()
@mod.manual_wrapper()
@function_with_aio(_ManualFunctionWithAio)
def manual_function(value: int) -> str:
    return f"sync:{value}"


class _ManualMethodWithAio(MethodWithAio):
    async def aio(self, value: int) -> str:
        return f"aio:{value}"


@mod.wrap_class()
class Client:
    @mod.manual_wrapper()
    @method_with_aio(_ManualMethodWithAio)
    def manual_method(self, value: int) -> str:
        return f"sync:{value}"
```

The generated code re-exports these entries directly instead of parsing signatures and emitting a new wrapper body. At module scope that becomes a simple alias like `manual_function = mylib._impl.manual_function`; inside wrapped classes, the marked attribute is copied into the emitted wrapper class unchanged. The same pattern also works for re-exporting a whole class directly when that class is intentionally not generated as a wrapper:

```python
from mylib.synchronicity2 import Module

mod = Module("mylib.api")


@mod.wrap_class()
@mod.manual_wrapper()
class ExistingPublicType: ...
```

As a rule of thumb, prefer `manual_export(...)` for plain module aliases. Use `manual_wrapper()` for descriptor-managed functions, methods, class attributes, or pre-existing public classes that need to participate in wrapper generation without a generated wrapper body.

Generated wrapper classes normally include public methods, selected protocol dunders, and manually registered underscored attributes. If an implementation class intentionally exposes single-underscore methods as part of its public wrapper API, opt in at the class level:

```python
import collections.abc

from mylib.synchronicity2 import Module

mod = Module("mylib.api")


@mod.wrap_class(include_underscored_methods=True)
class Client:
    async def _logs(self) -> collections.abc.AsyncGenerator[str, None]: ...
```

This generates normal sync/async wrappers for single-underscore instance methods and classmethods like `_logs`. Private staticmethods, double-underscore protocol methods, properties, and class attributes keep their usual explicit handling; this option is not a blanket export of all class internals.

### (Optional) Vendor the synchronicity2 runtime
There is a built-in option to vendor the runtime part of synchronicity2 into your package, which lets you use it in your published packages without introducing a runtime dependency on the pypi `synchronicity2` package.

**Creating the vendored modules**
To vendor the runtime, decide where in your package hierarchy you want it (`mylib.synchronicity2` in the following example) and point to the root dir of your package with the following command:
```bash
synchronicity2 vendor mylib.synchronicity2 -o src/
```

**Generate wrappers**
To make generated wrappers use the vendored synchronicity2 instead of the top level `synchronicity2` packatge, pass the `--runtime-package` option to the `synchronicity2 wrapper` command whenever you generate wrappers. All other options stay the same.

   ```bash
   synchronicity2 wrappers --runtime-package mylib.synchronicity2 --synchronizer-module mylib._synchronizer \
     -m mylib._weather_impl  -o src
   ```

The package contains all public top level exports from the synchronicity2 package, so any custom implementation code should use `from mylib.synchronicity2 import ...` to import helpers such as `FunctionWithAio`.

If you don't vendor synchronicity2, generated wrappers will depend on `synchronicity2` to be an importable package - it's functionally equivalent to `--runtime-package synchronicity2` (this can be useful for development of the synchronicity2 runtime itself since it won't require you to rerun vendoring on every change to the runtime).

It's recommended to create the vendoring source tree once and then only re-run this when you upgrade synchronicity2 and want to refresh the copy.


## Supported wrapper types

### Functions

Generated function wrappers expose a dual interface:

- Regular calls are synchronous and block until the async implementation finishes.
- `.aio(...)` exposes the async variant for awaitable functions and methods.

```python
from mylib.weather import get_temperature

assert get_temperature("Stockholm") == 20.0


async def main() -> None:
    assert await get_temperature.aio("Stockholm") == 20.0
```

### Classes and methods

```python
from mylib.weather import WeatherClient

client = WeatherClient("Stockholm")
assert client.current() == 21.0


async def main() -> None:
    client = WeatherClient("Stockholm")
    assert await client.current.aio() == 21.0
```

### Async streams (generators, iterators, and iterables)

The compiler uses your **return annotations** to tell async **generators**, **iterators**, and **iterables** apart (`AsyncGenerator[...]`, `AsyncIterator[...]`, `AsyncIterable[...]`, and class methods that implement `__aiter__` / `__anext__`). Use the right shapes in implementation code and generation will produce matching dual sync/async usage.

Here is a **one-way async generator** function from the weather example—sync callers use an ordinary `for` loop; async callers use `.aio()` with `async for`:

```python
import asyncio

from mylib.weather import stream_temperature_readings

# sync consumption
for item in stream_temperature_readings():
    print(item)


async def main() -> None:
    # async consumption:
    async for item in stream_temperature_readings.aio():
        print(item)
```

**Two-way** async generators annotated as `AsyncGenerator[YieldType, SendType]` also get first-class wrappers: the sync side supports `.send(...)` and `.close()`, the async side from `.aio(...)` supports `.asend(...)` and `.aclose()`, and cleanup is forwarded so closing the wrapper waits for async generator finalization. Wrapped classes in yield positions are translated to public wrappers, while wrapped values sent back into the generator are translated to implementation instances.

Synchronous `Generator[YieldType, SendType, ReturnType]` annotations preserve and translate all three type positions. `Iterator[ItemType]` remains a distinct one-way iterator rather than being widened to `Generator[ItemType, None, None]`.

### Detailed support details

The current codebase and tests cover:

- async functions and functions returning typed `Awaitable[...]`, exposed with `.aio(...)`
- wrapper-side translation of wrapped classes in annotated arguments and return values, including common container shapes like `list[...]`, `tuple[...]`, and `Optional[...]`
- async generators, including two-way generators with `send`/`asend` and cleanup via `close`/`aclose`
- synchronous generators with translated yield, send, and terminal return values
- sync and async iteration over wrapped async iterables and iterators
- async context managers, including direct `__aenter__`/`__aexit__` wrappers and functions or methods returning async context manager values
- wrapped classes with public instance methods
- wrapped `classmethod` and `staticmethod`
- sync methods on wrapped classes
- constructor argument translation for wrapped types
- cross-module wrapper generation
- wrapped class inheritance, including mirrored generic bases and wrapped base classes
- generic classes and functions with type variables
- generated properties from annotated public attributes
- type checking of generated wrappers and support files with pyright as part of the integration test suite

When a wrapped class inherits from another wrapped class, the generated wrapper preserves that public inheritance structure. In other words, if your implementation has `WrappedSub(WrappedBase)`, the generated public API also has `WrappedSub(WrappedBase)`, inherited wrapped methods stay available on the subclass, and `isinstance(sub, WrappedBase)` works on the public wrapper side.

## CLI usage

The package installs a `synchronicity2` CLI with two subcommands: `vendor` (copy runtime into your tree) and `wrappers` (generate public modules from `Module`-registered implementation code).

```bash
synchronicity2 wrappers --synchronizer-module mylib._synchronizer \
  -m mylib._impl --runtime-package mylib.synchronicity2 -o src
```

`wrappers` options:

- `-m/--module`: import module containing one or more `Module` objects; repeatable
- `--preload-module`: import a module before implementation modules; repeatable and intended for legacy
  Synchronicity 1 migrations where imports register wrappers
- `--synchronizer-module`: required qualified path for the generated module that owns the `Synchronizer` instance
  shared by every wrapper in this invocation
- `--synchronicity1-synchronizer`: optional importable S1 synchronizer as `MODULE:ATTRIBUTE`
- `-o/--output-dir`: root directory for generated files; paths mirror the `Module` target (e.g. `Module("mylib.weather")` with `-o src` writes `src/mylib/weather.py`)
- `--stdout`: print generated modules to stdout instead of writing files
- `--ruff`: run `ruff check --fix` and `ruff format` on generated output
- `--runtime-package`: dotted import path for generated imports of `types` / `descriptor` / `synchronizer` (default: `synchronicity2`; use your vendored package when shipping a self-contained wheel)

The same commands work as `python -m synchronicity2.codegen wrappers ...` and `python -m synchronicity2.codegen vendor ...`.

## Low-level runtime API

`Synchronizer` is part of the public package API and can be used for delegating coroutine calls to the synchronicity event loop on a one-off basis.

The philosophy is:

- library implementation code should usually use `Module` declarative wrappers
- generated wrapper code uses a single `Synchronizer` instance to coordinate around a single thread/event loop
- direct `Synchronizer` usage is mainly for advanced/manual cases and internal runtime behavior

## Why this exists

Library authors often want to maintain a single implementation of their API, and async code is usually the most convenient way to implement libraries with high concurrency. It composes well with network and I/O heavy code, works naturally for streaming results, and handles long-lived resources like connections or sessions cleanly.

The problem is that many consumers of a library still want a synchronous interface. They may be writing scripts, working in mostly blocking codebases, or just not want to structure their application around `asyncio` to call one library.

You can build that synchronous API manually as a wrapper over the async one, but it is inconvenient:

- any changes to the underlying async interface would require changes in the sync interface - you have to write and maintain two interfaces
- classes, methods, and translated argument and return types add a lot of boilerplate
- generators, iterators, and long-lived async state make wrapping logic much more complicated
- there are many foot guns on how to handle various interactions across the sync -> async boundary

Simple `asyncio.run()` wrappers are fine for one-off calls, but they are not powerful enough when you need:

- persistent async state across calls (network connections, synchronization primitives etc.)
- async generators, iterators, context manager support
- one implementation that supports both sync and async consumers seemlessly in the same application

## Design decisions

### Combined functions for sync/async usage

- The decision to use `.aio()` "dual" functions was made largely to avoid having to have two different types for every wrapped class (one with async method and one with sync methods). The notable downside to using "dual" functions instead of distinc types is that during async usage it becomes very easy to accidentally use blocking wrappers in async code - causing event loop blockage.
Two other variant were explored a long time ago:
- It was considered to auto infer at runtime if functions run inside of event loops and use that info to either run the underlying function blockingly or return an awaitable. This is bad for several reasons - it can't be statically typed, and sync functions that run inside event loops wouldn't be able to await the async awaitable returned anyways ("event loop already running")
- It was also considered to use distinct async and blocking types. The main disadvantages of this is:
  - Usage of both sync and async SDKs in the same application becomes clunky
  - Serialization of wrapped objects somewhat enforces that receiver also use sync or async interface (i.e. "mixed usage in distributed applications")
  - Namespace bloat of the top level namespace (could be mitigated if we make a different top level package for aio)

  Arguably these aren't very strong disadvantages and they can be mitigated by having conversion utilities, so I think we might want to consider distinct types as an alternative wrapper syntax going forward.

### Using the synchronizer event loop for async usage
For async usage, we still use the synchronizer event loop instead of just running it directly on the caller's event loop. This has a couple of advantages:
* Event loop blockage in the user's code won't break library code since the library code runs on its own thread/event loop
* Mixed usage of sync and async APIs in the same applications interoperate well since the library code runs on the same event loop so underlying eventloop specific code can be reused (e.g. network connections)

The big disadvantage is that it introduces additional call stack height and thread synchronization primitives which gives worse performance and traceback readability.

### Iterator syntax

Use `.aio()` to select the async side of a callable async generator: `async for x in async_generator_func.aio(): ...`. Once you have an iterable or iterator object, iterate it directly with `async for x in iterable: ...`.

## Practical gotchas

- `.aio(...)` runs implementation code on the generated module's shared synchronizer loop in a background thread, not on the caller's current event loop.
- Wrapped objects are translated back to implementation objects when passed into wrapped functions, methods, and constructors.
- Wrapper identity is preserved across that boundary: if the same implementation object comes back out, you get the same wrapper instance back.
- Async iterables and async iterators preserve their usual semantics. Iterator-like wrappers can be single-use or stateful, while iterable-like wrappers can be iterated repeatedly.
- Wrapped class instances are proxies around implementation instances, so ordinary implementation attributes are not part of the public wrapper unless exposed intentionally.
- Sync calls cross a thread boundary into the synchronizer loop, so there is some dispatch overhead compared with calling the raw implementation directly.
- Generated modules import the implementation modules at runtime, so generation does not make the implementation code disposable.
- Generated wrapper modules import runtime pieces like `synchronicity2.synchronizer` by default, or your vendored path (e.g. `mylib.synchronicity2.synchronizer`) when using `--runtime-package`. They do not need `synchronicity2.codegen` at runtime.
- Fallback runtime wrapping for dynamically typed returns is opt-in, not automatic. If something dynamic like `__getattr__` or `__get__` should wrap a known synchronized type but otherwise fall back to opaque values, annotate it as a union such as `_PartialFunction | typing.Any` or `Payload | typing.Any`. Synchronicity will honor the wrapped union arm at runtime and treat the `Any` arm as the no-translation fallback.

## Current limitations

Some design ideas are still future work. In particular, the current implementation does not yet aim to cover every async protocol automatically.

Known gaps worth keeping in mind:

- unwrapped base classes are not reflected in generated wrapper inheritance

Callable support is partially implemented and covered by tests for the following shapes:

- callable-valued parameters are passed through unchanged at runtime. Generated wrapper annotations preserve the callable shape, including `Coroutine[...]` / `Awaitable[...]` designators, but nested synchronized types inside those callable signatures remain implementation-facing types.
- returned non-async callables like `Callable[..., Sequence[Node]]` and `Callable[P, list[T]]`, where wrapped implementation results are translated back to wrapper values

The currently tested scope is specifically non-async callable return values. Async callable return values and more advanced callable protocol shapes are not yet a documented or tested feature boundary.

For example, if your implementation looks roughly like:

```python notest
class UnwrappedBase:
    def unwrapped_method(self) -> bool:
        return True


@mod.wrap_class()
class WrappedBase(UnwrappedBase):
    async def wrapped_method(self) -> list[int]:
        return []
```

then the generated wrapper exposes `wrapped_method`, but it does not expose `unwrapped_method` on the public wrapper class by default.

This is intentional. Wrapper instances are proxies around implementation instances rather than actual subclasses of the implementation classes, so an unwrapped base method would run with a different `self` object than the implementation class expects. In practice, that means inherited unwrapped code could observe or mutate different attributes than it would on the real implementation instance. Wrapped bases are safe to mirror because their public methods are re-generated against the wrapper model; unwrapped bases are not mirrored automatically for that reason.

## Typing caveats

The generated APIs are tested with pyright, including consumer-side usage examples, but some advanced typing forms still have rough edges.

Callable typing is covered by pyright-based integration tests for:
- decorator-style callable parameters whose signature must be preserved in the public wrapper API
- passthrough callable parameters, including coroutine-returning callback annotations
- returned non-async callables whose results contain synchronized types

Remaining caveats are mainly around more advanced or not-yet-supported callable forms, especially async callable return values.

## Runtime architecture

Codegen creates one synchronizer module per invocation. Generated wrappers import its `synchronizer` attribute, relying
on Python's module cache and import lock to provide one shared instance per interpreter.

That runtime component:

- owns a dedicated event loop in a background thread
- runs async work for sync callers
- also provides the async `.aio(...)` path on the same isolated loop
- preserves wrapper identity for wrapped classes via `_from_impl(...)`

Implementation modules should usually not need to import `Synchronizer` directly.

## Migrating from 0.x

The main architectural difference from Synchronicity `0.x` is that `2.x` is code-generation-based whereas `0.x` used runtime determination of how to proxy each call. The interface from a user of a library is still 99% backwards compatible, but the authoring process is slightly changed.

In `0.x`, the typical model was:

- create a `Synchronizer` in library code
- wrap functions and classes directly using a `@synchronizer.wrap()` decorator. The resulting entity could be used directly at runtime - no preprocessing step needed. This is the entity that was exposed as the public API.
- optionally run a build-time type-stub generation step to give the wrappers static types

In `2.x`, the typical model is:

- keep implementation modules as normal async Python with simple markers for what to translate
- register functions and classes with `Module` manifests
- generate wrapper source files ahead of time
- publish or import those generated modules as the public API

For gradual migrations from synchronicity 0.x ("synchronicity1"), codegen can be configured to respect synchronicity1 type wrapping/unwrapping and use an existing synchronicity1 event loop for interoperability.

```bash
synchronicity2 wrappers \
  --synchronizer-module my_library._synchronizer \
  --preload-module my_library.legacy_api \
  --synchronicity1-synchronizer my_library._runtime:legacy_synchronizer \
  -m my_library._new_impl \
  -o src
```

The Synchronicity 1 synchronizer remains responsible for starting and stopping its thread and event loop.
Synchronicity2 schedules its own coroutines on *that* loop and retains its own generated translation behavior. 

Synchronicity2 registers each generated implementation/wrapper type pair with Synchronicity 1 as an external wrapper, and synchronicity2 in turn detects any synchronicity1 types in type annotations and delegates wrapping/unwrapping to the synchronicity1 synchronizer when needed.

For Synchronicity 1 types to be detected during codegen where the wrapper isn't necessarily declared in the same file as the implementation, it's important to point out which modules contain wrapper type declarations. This is done by providing `--preload-module` optoins.

Migration considerations:

- Separate the modules where implementation is defined from the ones where the "public" wrappers are defined
- Replace `@synchronizer.wrap` authoring patterns with `@module.wrap_function` and `@module.wrap_class`.
- Add or tighten type annotations if older code relied on runtime inspection; the new compiler uses annotations heavily.
- If you previously documented direct `Synchronizer` usage as the primary user-facing pattern, update examples to show generated modules instead.
- Treat `Synchronizer` as a lower-level primitive that still exists, but is no longer the main recommended entry point for library authors.

## Development / Contribution

See [AGENTS.md](AGENTS.md) for development practices and [ARCHITECTURE.md](ARCHITECTURE.md) for the
runtime and code-generation design.

## TODO

- Revisit the seeming inconsistency between callable async generators requiring `.aio()` and iterable or iterator objects supporting direct async iteration. A more uniform API would be easier to explain and use.
