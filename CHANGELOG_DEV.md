# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

Draft releases notes should be added as part of the PR introducing the change. During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- New Environment names (via `modal environment create`, `modal environment update --set-name`, or `modal.Environment.objects.create`) can no longer contain periods.
- Improved the performance of Sandbox filesystem writes (`write_bytes`, `write_text`, `copy_from_local`).

- Added an `include_terminated` parameter to `Sandbox.from_name()`. If set to `True` and no Sandbox with the name is currently running, the most recent terminated Sandbox will be returned instead.
- Added `--since` and `--until` to `modal function calls` and `modal server requests` to fetch all records in a time range.

## JS

- Added `client.Functions.FromID()` to look up a Function by ID.
- Added [`Function_.stats()`](/docs/sdk/js/latest/Function#stats) to retrieve historical Function statistics over a time range.
- Improved the performance of Sandbox filesystem writes (`writeBytes`, `writeText`, `copyFromLocal`).
- Added an `includeTerminated` option to `client.sandboxes.fromName()`. If set to `true` and no Sandbox with the name is currently running, the most recent terminated Sandbox will be returned instead.

## Go

- Added `client.Functions.FromID()` to look up a Function by ID.
- Added [`Function.Stats()`](/docs/sdk/go/latest/Function#stats)  to retrieve historical Function statistics over a time range.
- Improved the performance of Sandbox filesystem writes (`WriteBytes`, `WriteText`, `CopyFromLocal`).
- Added an `IncludeTerminated` field to `SandboxFromNameParams`. If set to `true` and no Sandbox with the name is currently running, `Sandboxes.FromName` returns the most recent terminated Sandbox instead.
