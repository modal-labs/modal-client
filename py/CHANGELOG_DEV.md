# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Updated `Function.update_autoscaler()` and `Cls.update_autoscaler()` to return a `FunctionAutoscalerSettings` dataclass reflecting the current settings after applying the update.
- Updated `Server.update_autoscaler()` to return a `ServerAutoscalerSettings` dataclass reflecting the current settings after applying the update.
- Added `Image.logs`(/docs/sdk/py/latest/Image#logs) namespace to retrieve image build logs. This is useful for debugging image build failures. Use `Image.logs.fetch()` to fetch logs for multiple build layers and `Image.logs.tail()` to retrieve the latest build entries.
