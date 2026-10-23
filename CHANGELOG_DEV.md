# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

Draft releases notes should be added as part of the PR introducing the change. During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- Added the `Workspace.member_tokens` manager, along with the `modal workspace member-tokens` CLI, which allows users to list and delete tokens associated with Workspace Members.
- Fixed `modal.Image.from_dockerfile(..., force_build=True)` to rebuild the Dockerfile's own steps instead of reusing them from the cache.
- Added an `App.stop()` method.
- Added a `--set-image-builder-version` option to `modal environment update` for overriding the image builder version of a single Environment, and an image builder version column to `modal environment list`.

## JS

- Added `client.environments.fromContext()` to look up the Environment defined by the local configuration, or the server's default Environment.
- Added an `App.stop()` method.

## Go

- Added `client.Environments.FromContext()` to look up the Environment defined by the local configuration, or the server's default Environment.
- Added an `App.Stop()` method.
