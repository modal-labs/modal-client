# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Added a modal-native Image registry. Use `modal.Image.publish()` to publish images under a {name}:{tag} alias and `modal.Image.from_name()` for later referencing said image without invalidating/triggering a build.
- We've added a new `modal skills` CLI for installing a foundational Modal agent skill (`modal skills install`). This skill can (and should) be kept up to date over time with `modal skills update`.
- It's now possible to restrict *outbound* Sandbox traffic to a set of domains by setting `outbound_domain_allowlist=[...]` in `modal.Sandbox.create()`. When set, the Sandbox can only reach the listed domains.
- `Sandbox.snapshot_filesystem` and `Sandbox.snapshot_directory` now accept an explicit `ttl` keyword argument (in seconds) that controls how long the resulting Image is retained. Both methods default to `ttl=30 * 24 * 3600` (30 days). This is a change of default for `snapshot_filesystem` which previously kept Images indefinitely. Pass an explicit `ttl=None` to opt out of expiry.
- `Sandbox.snapshot_directory` now also accepts a `timeout` keyword argument (default `55` seconds), bringing it to parity with `snapshot_filesystem`. If the snapshot does not return within that window, `modal.exception.TimeoutError` is raised. The timeout can be set arbitrarily high to preserve the old behavior of not timing out.
- `sandbox.filesystem.watch(path)` watches a remote path in the Sandbox for filesystem changes. This replaces the alpha `modal.Sandbox.watch` method.
- Added version-pinned lookups in `modal.Function.from_name` and `modal.Cls.from_name` via the version parameter. Function invocations made from within a version-pinned function call now use the caller’s version.
- CLI `--json` output now uses snake_case keys (e.g. `"created_at"` instead of `"Created at"`) for programmatic consumption.
