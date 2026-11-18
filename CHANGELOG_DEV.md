# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

Draft releases notes should be added as part of the PR introducing the change. During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- Added `Workspace.audit_logs.fetch()` and `Workspace.audit_logs.count()` to read and count workspace audit events from the SDK. Counting requires an explicit `since` datetime. The `modal workspace audit-logs` CLI streams full events as newline-delimited JSON.
- Added `modal workspace audit-logs --csv` to export workspace audit events as CSV.
- `Sandbox.wait_until_ready()` now works without a readiness probe, returning when the Sandbox has started.
- Fixed a bug where `.map()`, `.starmap()`, and `.for_each()` could hang indefinitely when called with no inputs.
- Updated `modal workspace member-tokens delete` and `Workspace.member_tokens.delete()` such that they can now take in multiple token IDs for batch deletion.

## JS

## Go
