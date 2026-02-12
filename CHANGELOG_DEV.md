# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Added `Queue.from_id()` and `Dict.from_id()` methods to reference a Queue or Dict by its object id.
- We've added a new `modal billing report` CLI.
- Modal's async usage warnings are now enabled by default. These warnings will fire when using a [blocking interface on a Modal object](https://modal.com/docs/guide/async) in an async context. We've aimed to provide detailed and actionable suggestions for how to modify the code, which makes the warnings verbose. While we recommend addressing any warnings that pop up, as they can point to significant performance issues or bugs, we also provide a configuration option to disable them (`MODAL_ASYNC_WARNINGS=0` or `async_warnings = false` in the `.modal.toml`). Please report any apparent false positives or incorrect suggested fixes.
- Fixed a bug where the `state` contents could leak between requests when using `@modal.asgi_app`.
