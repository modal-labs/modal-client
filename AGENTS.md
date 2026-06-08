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
