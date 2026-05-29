# Modal SDK codebases

This directory contains codebases for Modal's Python, JS, and Go SDKs. It also contains protobuf definitions for the public gRPC API.

The contents of this directory are mirrored to a *public* GitHub repository: https://github.com/modal-labs/modal-client.

## Key Development Considerations

Any `inv` commands given can be run from the Modal monorepo root as `inv -r client/ ...`.

**Protocol Buffers**: Proto files must be organized into sections ordered as: `import`, `enum`, `message`, `service`. Within each section, definitions must be lexicographically sorted by name. Verify with `inv lint-protos`.
