# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

Draft releases notes should be added as part of the PR introducing the change. During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- Improved reliability of Sticky Session lifecycle operations by retrying transient failures and allowing session starts to wait up to 25 minutes.
- Added `modal endpoint info` command for displaying information such as an endpoint's deployment status, URL, and model id. We've also added `modal endpoint stats` to inspect performance metrics for an Endpoint over a selected time window and `modal endpoint logs` for displaying logs for an Endpoint.
- Added `container_total_count` to the output returned by the `stats` command of Function, Server, and Endpoint CLI groups. This field is also returned in the `.stats` method on `Server` and `Function` objects.

## JS

- Added `client.Functions.FromID()` to look up a Function by ID.

## Go

- Added `client.Functions.FromID` to look up a Function by ID.
