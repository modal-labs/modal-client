# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- We've added a new `modal skills` CLI for installing a foundational Modal agent skill (`modal skills install`). This skill can (and should) be kept up to date over time with `modal skills update`.
- `Sandbox.snapshot_filesystem` and `Sandbox.snapshot_directory` now accept an explicit `ttl` keyword argument (in seconds) that controls how long the resulting Image is retained. Both methods default to `ttl=30 * 24 * 3600` (30 days). This is a change of default for `snapshot_filesystem` which previously kept Images indefinitely. Pass an explicit `ttl=None` to opt out of expiry.
- `Sandbox.snapshot_directory` now also accepts a `timeout` keyword argument (default `55` seconds), bringing it to parity with `snapshot_filesystem`. If the snapshot does not return within that window, `modal.exception.TimeoutError` is raised. The timeout can be set arbitrarily high to preserve the old behavior of not timing out.
