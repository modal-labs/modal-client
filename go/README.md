# Modal Go SDK

[![Go Reference Documentation](https://pkg.go.dev/badge/github.com/modal-labs/modal-client/go)](https://pkg.go.dev/github.com/modal-labs/modal-client/go)

The [Modal](https://modal.com/) Go SDK provides convenient, on-demand access to
serverless cloud compute on Modal from golang projects. Use it to safely run
arbitrary code in Modal Sandboxes, call Modal Functions, and interact with Modal
resources.

We're approaching feature parity with the main
[Modal Python SDK](https://github.com/modal-labs/modal-client), although
defining Modal Functions will likely remain exclusive to Python.

## Installation

Install the latest version:

```bash
go get -u github.com/modal-labs/modal-client/go
```

Import in your application:

```go
import "github.com/modal-labs/modal-client/go"
```

Go package: https://pkg.go.dev/github.com/modal-labs/modal-client/go

### Authenticating with Modal

You also need to authenticate with Modal (see
[Getting started](https://modal.com/docs/guide#getting-started)). Either sign in
with the Modal CLI using `pip install modal && modal setup`, or in machine
environments set these environment variables:

```bash
# Replace these with your actual token!
export MODAL_TOKEN_ID=ak-NOTAREALTOKENSTRINGXYZ
export MODAL_TOKEN_SECRET=as-FAKESECRETSTRINGABCDEF
```

### Telemetry and Observability

The Modal Go SDK supports custom gRPC interceptors for telemetry, tracing, and
observability. You can add custom unary and stream interceptors to measure API
call latency, trace requests, and integrate with observability tools like
OpenTelemetry, DataDog, and others. See the
[telemetry example](./examples/telemetry/main.go) for more details.

## Requirements

Go 1.23 or later.

## Documentation

See the main [Modal documentation](https://modal.com/docs) and
[user guides](https://modal.com/docs/guide) for high-level overviews. For
details, see the
[API reference documentation for for Go](https://pkg.go.dev/github.com/modal-labs/modal-client/go#section-documentation).

We also provide a number of examples:

- [Call a deployed Function](./examples/function-call/main.go)
- [Spawn a deployed Function](./examples/function-spawn/main.go)
- [Call a deployed Cls](./examples/cls-call/main.go)
- [Call a deployed Cls, and override its options](./examples/cls-call-with-options/main.go)
- [Create a Sandbox](./examples/sandbox/main.go)
- [Create a named Sandbox](./examples/sandbox-named/main.go)
- [Create a Sandbox with GPU](./examples/sandbox-gpu/main.go)
- [Create a Sandbox using a private image from AWS ECR](./examples/sandbox-private-image/main.go)
- [Take a snapshot of the filesystem of a Sandbox](./examples/sandbox-filesystem-snapshot/main.go)
- [Snapshot a directory, and mount it in a running Sandbox](./examples/sandbox-directory-snapshot/main.go)
- [Execute Sandbox commands](./examples/sandbox-exec/main.go)
- [Running a coding agent in a Sandbox](./examples/sandbox-agent/main.go)
- [Check the status and exit code of a Sandbox](./examples/sandbox-poll/main.go)
- [Access Sandbox filesystem](./examples/sandbox-filesystem/main.go)
- [Expose ports on a Sandbox using Tunnels](./examples/sandbox-tunnels/main.go)
- [Create connect tokens for a Sandbox](./examples/sandbox-connect-token/main.go)
- [Include Secrets in Sandbox](./examples/sandbox-secrets/main.go)
- [Mount a Volume to a Sandbox](./examples/sandbox-volume/main.go), and same but
  [with an ephemeral Volume](./examples/sandbox-volume-ephemeral/main.go)
- [Mount a cloud bucket to a Sandbox](./examples/sandbox-cloud-bucket/main.go)
- [Eagerly build an Image for a Sandbox](./examples/sandbox-prewarm/main.go)
- [Building custom Images](./examples/image-building/main.go)
- [Add telemetry and tracing with custom interceptors](./examples/telemetry/main.go)

## Support

For usage questions and other support, please reach out on the
[Modal Community Slack](https://modal.com/slack).
