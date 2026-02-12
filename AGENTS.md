# Rules for working with LLMs on the Modal client codebase

This file provides guidance to LLMs when working with code in this repository.

## Common Development Commands

**Code Quality:**

Before making a pull request, the following checks must pass:

- `inv lint --fix` - Run ruff with auto-fix.
- `inv lint-protos` - Lint protocol buffer definitions.
- `inv type-check` - Run mypy and pyright type checking.
- `inv check-copyright --fix` - Ensure that all files have a copyright header.

**Development Tasks:**

- `inv protoc` - Compile protocol buffer files.
- `inv lint-protos` - Lint protocol buffer definitions.
- `inv type-stubs` - Regenerate the *.pyi type stub files using the synchronicity library.

**Running tests:**
- `inv test` - Run all tests
- `inv test --pytest-args "test_file.py::specific_test` - Pass arguments to pytest directly, e.g. to run specific tests.

## Code Quality Standards

**Linting**: ruff configured for 120 character line length with import sorting.

**Type Checking**: Both MyPy and Pyright used for different file sets (see `inv type-check` in `tasks.py`).

**Formatting**: ruff handles both linting and formatting.

**Copyright**: All Python files must start with `# Copyright Modal Labs {year}` (see `inv check-copyright --fix` in `tasks.py`).

**Imports**: Follow isort configuration with Modal packages as first-party.

## Key Development Considerations

**Synchronicity Wrappers**: When adding new async methods, ensure they're properly wrapped for sync API generation.

**Protocol Buffers**: Changes to `.proto` files require running `inv protoc` to regenerate Python stubs.

**Type stubs**: All `*.pyi` files are auto-generated and should never be manually edited.

**Object Lifecycle**: New Modal resource types should inherit from `_Object` and follow established patterns.

**Error Handling**: Use Modal's exception hierarchy and ensure proper error propagation between client/server.

**Testing Async Code**: Use pytest-asyncio and ensure proper async test setup in conftest.py.

**`.from_...` Constructors**: All new `.from_id()`, `.from_name()`, and similar constructors on Modal objects should be **non-async** and use `_from_loader` for deferred/lazy loading. This is important because these constructors are often called at module-level (global scope) when declaring references to remote assets. Making them async would cause blocking I/O during module import, which breaks many use cases. The actual RPC call should happen inside the `_load` function passed to `_from_loader`, which will be invoked lazily when the object is first used.
