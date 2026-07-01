# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**
- `modal.Sandbox.reload_volumes` now blocks until the Volumes have been reloaded, bounded by a new `timeout` argument (55 seconds by default). If the reload does not complete within `timeout`, `modal.exception.TimeoutError` is raised.
- `modal profile activate` will now print the active environment associated with that profile.
- `Function.with_options` now accepts a `routing_region` argument to override the region the Function's inputs and outputs will be routed through.
- Added `modal workspace settings list` to view workspace settings and `modal workspace settings set <setting> <value>` to update them (e.g. `modal workspace settings set default-environment prod`). Valid settings are `default-environment` and `image-builder-version`. The corresponding Python API is available via `Workspace.from_context().settings`, which exposes `list()` and `set(name, value)`.
- `modal container stop` now accepts a `--graceful` flag. With it, the container stops fetching new inputs but finishes the inputs it is currently running before exiting, instead of having them cancelled and rescheduled. Graceful stops are only supported for containers running a Modal Function or Modal Server.
