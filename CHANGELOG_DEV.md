# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

Draft releases notes should be added as part of the PR introducing the change. During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- Fixed a bug where `.map()`, `.starmap()`, and `.for_each()` could hang indefinitely when called with no inputs.

## JS

- Added `App.stats()` to report high-level Function, Server, and Sandbox stats for an App.
- Added `client.environments.fromContext()` to look up the Environment defined by the local configuration, or the server's default Environment.
- Added an `App.stop()` method.
- Sandboxes no longer set the `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `BLIS_NUM_THREADS`, `ORT_INTRA_OP_NUM_THREADS` and `ORT_INTER_OP_NUM_THREADS` environment variables by default.
- Added the `_Function.info()` method.
- `Image.build()` now throws `NotFoundError` instead of a raw gRPC error when built on top of an Image that doesn't exist.
- Added `app.sandboxes.stats()` to report the cost of an App's Sandboxes over a time range, optionally grouped by tag values. Costs are returned as decimal strings.

## Go

- Added an `App.Stats()` method to report high-level Function, Server, and Sandbox stats for an App.
- Added `client.Environments.FromContext()` to look up the Environment defined by the local configuration, or the server's default Environment.
- Added an `App.Stop()` method.
- Sandboxes no longer set the `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `BLIS_NUM_THREADS`, `ORT_INTRA_OP_NUM_THREADS` and `ORT_INTER_OP_NUM_THREADS` environment variables by default.
- Added a `Function.Info()` method.
- `Image.Build()` now returns `NotFoundError` instead of a raw gRPC error when built on top of an Image that doesn't exist.
- Added `App.Sandboxes.Stats()` to report the cost of an App's Sandboxes over a time range, optionally grouped by tag values. Costs are returned as decimal strings.
