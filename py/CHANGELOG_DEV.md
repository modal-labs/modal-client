# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**
- `modal profile activate` will now print the active environment associated with that profile.
- `Function.with_options` now accepts a `routing_region` argument to override the region the Function's inputs and outputs will be routed through.
- Added `modal workspace settings list` to view workspace settings and `modal workspace settings set <setting> <value>` to update them (e.g. `modal workspace settings set default-environment prod`). Valid settings are `default-environment` and `image-builder-version`. The corresponding Python API is available via `Workspace.from_context().settings`, which exposes `list()` and `set(name, value)`.
