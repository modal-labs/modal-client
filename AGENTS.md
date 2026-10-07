# Guidelines for coding agents

This directory contains codebases for Modal's Python, JS, and Go SDKs. It also
contains protobuf definitions for the public gRPC API.

The contents of this directory are mirrored to a _public_ GitHub repository:
https://github.com/modal-labs/modal-client.

## Language-specific SDKs

The Python SDK in `client/py` is the main Modal SDK, and considered to be the
reference implementation for other SDKs.

The JS and Go SDKs (in `client/js` and `client/go`, resp.) don't yet have all
the functionality of the Python SDK. We aim to keep JS and Go at feature parity
with each other, so new features should be added to both SDKs simultaneously. We
also aim to keep the JS and Go SDKs structurally similar, but make exceptions to
follow idiomatic language conventions.

## Key Development Considerations

Any `inv` commands given can be run from the Modal monorepo root as
`inv -r client/ ...`.

**Protocol Buffers**: Proto files must be organized into sections ordered as:
`import`, `enum`, `message`, `service`. Within each section, definitions must be
lexicographically sorted by name. Verify with `inv lint-protos`.

Certain implementation classes expose type-guarded versions of private attributes
as `@property` decorated accessors which are the attribute name suffixed with an `_`.
These should be used instead of the raw underlying attribute.

## Comments

Comments and docstrings must not refer to backend internals. This code is
public and is read by people with no view into backend/runtime implementation.
Where details are required to understand client behavior, describe the behavior
the client can observe instead.

## Changelog updates

The SDK source includes changelog files. These document public API or behavioral changes that are relevant to how end users interface with Modal. Examples where a changelog update is needed:

- New public API (or CLI) features are added, including new public objects/types, functions/methods, or parameters
- Changes to semantics that are relevant for user code, like different different default values, lazy->eager, blocking->async, etc.
- A stable feature is deprecated (starts issuing warnings) or a deprecation is enforced (the feature is removed)
- Fixes for bugs with significant implications for user code.
- Significant performance optimizations

Changelog updates are not needed in the following cases:

- Protobuf-only changes
- Changes to experimental APIs
- Changes to documentation or output
- Minor bug fixes or performance improvements

For multiple changes to the same feature within a single release cycle, edit an existing changelog entry rather than treating each update as a distinct change.

During development, updates are made to `client/CHANGELOG_DEV.md`. When making a release, changelog entries are moved to the language-specific changelogs (`client/py/CHANGELOG.md`, etc.) and edited for publication.

## Cursor Cloud specific instructions

This checkout is the public client repository. Work from `py/`, `js/`, and `go/` (there is no `client/` prefix here).

- Python development uses the Python 3.11 virtualenv at `py/.venv`. Activate it before `inv` commands (`source py/.venv/bin/activate` from the repo root, or run them from `py/`). `inv protoc` finds grpclib's protoc plugins on `PATH`, so the venv must be active. The system `python3` is not that interpreter.
- From `py/`: `inv lint`, `inv type-check`, and `inv test`. Python tests use an in-process mock gRPC server and do not need Modal credentials. The agent shell sets `TERM=dumb` and `NO_COLOR=1` and does not set `COLUMNS`. Rich then treats the console as a dumb, narrow terminal, and CLI selector tests plus wide table assertions fail. Run the suite with `env -u NO_COLOR -u FORCE_COLOR TERM=xterm-256color COLUMNS=200 inv test`.
- From `js/`: `npm ci`, `npm run check`, `npm run lint`, `npm run build`, and `npm test`. Node 22 is on `PATH`.
- `go` on `PATH` is Go 1.25 (`/usr/local/bin/go`). From `go/`, `go test ./...`.
- Many JS and Go tests call the live Modal API (the `libmodal-test-support` app). They need `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Tests that only use the in-repo gRPC mocks do not.
