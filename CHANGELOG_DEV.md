# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

Draft releases notes should be added as part of the PR introducing the change. During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- Added `modal image build -t NAME CONTEXT_DIR` to build and publish an Image from a Dockerfile. Use `--registry-secret` or `--registry-secret-from-env` to pull a base image from a private registry. Prefix NAME with an environment, e.g. `-t my-env/my-image`, to publish it into that environment.
- Added `modal endpoint library` to discover endpoint-compatible models.
- Functions restored from a memory snapshot now see current secret values in os.environ, instead of the secrets from when the snapshot was taken.

## JS

- Added `client.environments.fromContext()` to look up the Environment defined by the local configuration, or the server's default Environment.
- Added an `App.stop()` method.
- Added the `_Function.info()` method.

## Go

- Added `client.Environments.FromContext()` to look up the Environment defined by the local configuration, or the server's default Environment.
- Added an `App.Stop()` method.
