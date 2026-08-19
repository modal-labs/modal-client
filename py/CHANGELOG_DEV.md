# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Added `Sandbox.logs`(/docs/sdk/py/latest/Sandbox#logs) namespace to retrieve Sandbox entrypoint logs directly from the SDK. The namespace has two different methods, allowing you `fetch()` logs from a specific date/time range, or `tail()` the most recent logs.
- Added support for setting the default member Role when creating Restricted Environments through the Python SDK and CLI.
- The `modal` CLI now accepts a global `--profile` option for simpler ad hoc profile selection.
- `modal environment roles list --exclude-default` and
  `Environment.roles.list(exclude_default=True)` list only users and service users who have been
  directly assigned a role for the Environment.
