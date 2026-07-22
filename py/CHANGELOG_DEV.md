# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- Added a `--compute-region` option to `modal endpoint create` for selecting where Endpoint containers run independently from request routing.
- Added the `logs` namespace to [`Function`](/docs/sdk/py/latest/Function#logs), [`Server`](/docs/sdk/py/latest/Server#logs), and [`FunctionCall`](/docs/sdk/py/latest/FunctionCall#logs). You can now `stream()` logs as they arrive, `fetch()` all logs in a date range, or `tail()` the latest logs.
- The [`modal.Environment.members.list()`](/docs/sdk/py/latest/Environment#memberslist) method (and the [`modal environment members list`](/docs/cli/latest/environment#modal-environment-members-list) CLI) now returns every workspace member and service user together with their effective role in the Environment
- The `modal.Workspace.billing` and `modal.Environment.billing` views both have a new `.summary()` method which returns a succinct summary of billing information for a given interval.
- Added `modal billing summary` and `modal environment billing summary` as CLIs for the respective `modal.Workspace.billing.summary()` and `modal.Environment.billing.summary()` methods.
- The [`sandbox.filesystem.copy_from_local()`](/docs/sdk/py/latest/Sandbox#filesystemcopy_from_local), [`write_bytes()`](/docs/sdk/py/latest/Sandbox#filesystemwrite_bytes), and [`write_text()`](/docs/sdk/py/latest/Sandbox#filesystemwrite_text) methods now stream data to the Sandbox instead of issuing a request per chunk, making large writes ~2.5× faster.
