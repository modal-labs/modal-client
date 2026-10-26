# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

Draft releases notes should be added as part of the PR introducing the change. During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- Added the `Workspace.member_tokens` manager, along with the `modal workspace member-tokens` CLI, which allows users to list and delete tokens associated with Workspace Members.
- Fixed `modal.Image.from_dockerfile(..., force_build=True)` to rebuild the Dockerfile's own steps instead of reusing them from the cache.
- Added an `App.stop()` method.
- Added a `--set-image-builder-version` option to `modal environment update` for overriding the image builder version of a single Environment, and an image builder version column to `modal environment list`.
- Added implicit OAuth refresh token authentication through the `MODAL_OAUTH_REFRESH_TOKEN`, `MODAL_OAUTH_CLIENT_ID`, and either `MODAL_OAUTH_CLIENT_SECRET` or `MODAL_OAUTH_JWT_KEY` environment variables, with [`modal.Client.from_oauth_credentials()`](/docs/sdk/py/latest/Client#from_oauth_credentials) available for explicitly constructed clients. RSA private keys require the `modal[oauth]` extra.
- `Image.from_registry` now infers the kind of registry authentication from the keys in its `secret`, so a Secret with AWS ECR or GCP Artifact Registry credentials works without `Image.from_aws_ecr` or `Image.from_gcp_artifact_registry`.

## JS

- Added `client.environments.fromContext()` to look up the Environment defined by the local configuration, or the server's default Environment.
- Added an `App.stop()` method.

## Go

- Added `client.Environments.FromContext()` to look up the Environment defined by the local configuration, or the server's default Environment.
- Added an `App.Stop()` method.
