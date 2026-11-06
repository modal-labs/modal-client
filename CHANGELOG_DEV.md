# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

Draft releases notes should be added as part of the PR introducing the change. During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- Function parameters annotated as `X | None` / `Optional[X]` are now recorded with the type of `X` in the Function schema, so the dashboard's Try it form accepts input for them.
- Python 3.10 support is deprecated and will be removed in Modal 1.7.0. The client now warns when imported on Python 3.10 and when building an Image that uses Python 3.10.
- Fixed `modal shell --cmd` mangling quotes in the command when starting a new container; the command is now passed verbatim to `/bin/bash -c`.
- Added `modal image build -t NAME CONTEXT_DIR` to build and publish an Image from a Dockerfile. Use `--registry-secret` or `--registry-secret-from-env` to pull a base image from a private registry. Prefix NAME with an environment, e.g. `-t my-env/my-image`, to publish it into that environment.
- Added a `modal image pull` CLI command that pulls an Image from a container registry into Modal, e.g. `modal image pull docker.io/node:20 --name node`.
- Added `modal endpoint library` to discover endpoint-compatible models.
- Fixed `Image.cmd()`, `Image.entrypoint()`, and `Image.shell()` for arguments that contain double quotes or backslashes. Such arguments are now passed to the container exactly as written; previously a command like `Image.cmd(["sh", "-c", 'echo "hi"'])` failed when the container started.
- Functions restored from a memory snapshot now see current secret values in os.environ, instead of the secrets from when the snapshot was taken.
- Added `modal token new --delete-existing` and `modal token rotate` CLI commands to enable easy token rotation.
- `Function.from_id` will now accept Cls IDs. When called with a Cls ID, only the following APIs can be called: `.logs`, `.metrics`, `.info()`, `.stats()`, `.get_current_stats()`, `.get_web_url()` and `.autoscaler.settings()`. Use `modal.Cls.from_name(...)` to invoke all other methods.

## JS

- Added `client.environments.fromContext()` to look up the Environment defined by the local configuration, or the server's default Environment.
- Added an `App.stop()` method.
- Added the `_Function.info()` method.
- `Image.build()` now throws `NotFoundError` instead of a raw gRPC error when built on top of an Image that doesn't exist.

## Go

- Added `client.Environments.FromContext()` to look up the Environment defined by the local configuration, or the server's default Environment.
- Added an `App.Stop()` method.
- `Image.Build()` now returns `NotFoundError` instead of a raw gRPC error when built on top of an Image that doesn't exist.
