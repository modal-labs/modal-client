# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Allow `custom_domain` to be passed for a `Sandbox`. Note that sandbox custom domains work differently from other Modal custom domains and must currently be set up manually by Modal.
- The automatic CLI creation for `modal run` entrypoints now supports `Literal` type annotations, provided that the literal type contains either all string or all int values.
- Added `--timestamps` flag to `modal run`, `modal serve` and `modal deploy` to show timestamps for log output.
  ```bash
  modal run --timestamps my_app.py::my_function
  ```
- Added a `--timestamps` option to the `modal container logs` CLI.
- Adds experimental support for Python 3.14t. You can test Python 3.14t with the following image definition:
  ```python
  image = modal.Image.from_registry("debian:bookworm-slim", add_python="3.14t")
  ```
- Added a new `modal token info` CLI command to retrieve information about the credentials that are currently in use
- Correctly specify the minimum version for `grpclib` with Python 3.14.
- Improves client resource management when running `Sandbox.exec`.
