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
