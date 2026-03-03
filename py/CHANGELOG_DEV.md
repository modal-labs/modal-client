# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- We've added a `modal changelog` CLI for retrieving changelog entries with a flexible query interface (e.g. `modal changelog --since 1.2`, `modal changelog --newer`). We expect this can be a useful way to surface information about new features to coding agents.
- The new `modal.Secret.update` method allows you to programmatically modify the environment variables within a Secret.
- `modal.Function.get_current_stats` now returns a `num_running_inputs` denoting the number of running inputs.
