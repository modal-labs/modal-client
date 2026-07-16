---
description: Complete API reference for the Modal Go SDK.
---

# Go SDK Reference

This is the API reference for the [Modal](https://pkg.go.dev/github.com/modal-labs/modal-client/go) Go SDK.

## Installing

Install the library using `go get`:

```bash
go get github.com/modal-labs/modal-client/go@latest
```

Then you can import it in your code:

```go
import modal "github.com/modal-labs/modal-client/go"
```

Go 1.24.0 or later is required.

## Configuration

A Modal API token / secret pair is required for authentication. Credentials are read from the following sources in descending order of precedence:

- Parameters passed when constructing a new [`Client`](/docs/sdk/go/latest/Client)
- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` environment variables
- `token_id` and `token_secret` fields in the active profile of a `~/.modal.toml` file

API tokens can be created in the [workspace settings](/settings/tokens) on the Modal Dashboard.

Tokens can also be created and managed with the Modal [CLI](/docs/cli/latest). The CLI is packaged with the [Python SDK](/docs/sdk/py/latest), but it can also be installed as a standalone tool:

```bash
curl -LsSf uvx.sh/modal/install.sh | sh
```

## Objects

|  |  |
| --- | --- |
| [`App`](/docs/sdk/go/latest/App) | The main unit of deployment for code on Modal |
| [`Function`](/docs/sdk/go/latest/Function) | A serverless function backed by an autoscaling container pool |
| [`Image`](/docs/sdk/go/latest/Image) | An immutable representation of a container filesystem |
| [`Sandbox`](/docs/sdk/go/latest/Sandbox) | A process-like interface for restricted execution |
| [`Secret`](/docs/sdk/go/latest/Secret) | A secure reference to environment variables |
| [`Volume`](/docs/sdk/go/latest/Volume) | A mutable distributed storage device |

## Scope

The Go SDK is more limited in scope than Modal's [Python SDK](/docs/sdk/py/latest). Namely, Python remains the only supported Function runtime, so features for defining Functions are not available in Go. We do aim to support most interactions with remote Modal objects, including Functions, Sandboxes, Volumes, and others. Some in-scope features are not yet implemented; please get in touch if these are important to you.

## Usage

Instances of the Modal Object types are obtained through service methods on the [Client](/docs/sdk/go/latest/Client) type.

```go
mc, err := modal.NewClient()
if err != nil {
  log.Fatal(err)
}
defer mc.Close()
```

Interaction with instances of the Modal Objects happens via methods on those instances. All methods have an associated Params struct for passing optional arguments.

### Sandbox

The Go SDK aims to support the complete feature-set for Modal [Sandboxes](/docs/guide/sandboxes):

```go
app, err := mc.Apps.FromName(ctx, "sandbox-app", &modal.AppFromNameParams{CreateIfMissing: true})
if err != nil {
	log.Fatal(err)
}
image := mc.Images.FromRegistry("alpine:3.21", nil)

sb, err := mc.Sandboxes.Create(ctx, app, image, nil)
if err != nil {
	log.Fatal(err)
}
defer sb.Terminate(ctx, nil)

p, err := sb.Exec(ctx, []string{"echo", "Hello from a Sandbox!"}, nil)
if err != nil {
	log.Fatal(err)
}
output, err := io.ReadAll(p.Stdout)
if err != nil {
	log.Fatal(err)
}
fmt.Print(string(output))
```

Sandboxes accept many additional parameters at creation time for configuring their resources, lifecycle, restrictions, etc.:

```go
secret, err := mc.Secrets.FromName(ctx, "github-token", nil)
if err != nil {
	log.Fatal(err)
}

sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
	Secrets:                 []*modal.Secret{secret},
	Timeout:                 30 * time.Minute,
	CPU:                     2,
	MemoryMiB:               2048,
	OutboundDomainAllowlist: &modal.Allowlist{Entries: []string{"github.com", "*.githubusercontent.com"}},
})
if err != nil {
	log.Fatal(err)
}
defer sb.Terminate(ctx, nil)
```

### Function

Modal Functions in deployed Apps can be [looked up and invoked](/docs/guide/trigger-deployed-functions) from Go programs:

```go
f, err := mc.Functions.FromName(ctx, "my-app", "echo", nil)
if err != nil {
	log.Fatal(err)
}

result, err := f.Remote(ctx, []any{"Hello, Modal!"}, nil)

var remoteErr modal.RemoteError
if errors.As(err, &remoteErr) {
	log.Fatalf("remote call failed: %s", remoteErr.Exception)
} else if err != nil {
	log.Fatal(err)
}

response, ok := result.(string)
if !ok {
	log.Fatalf("expected a string result, got %T", result)
}
fmt.Println(response)
```

All Function invocation methods are available:

```go
fc, err := f.Spawn(ctx, []any{"Hello, Modal!"}, nil)
if err != nil {
	log.Fatal(err)
}

result, err := fc.Get(ctx, nil)
if err != nil {
	log.Fatal(err)
}
fmt.Println(result)
```

## Errors

Error types are documented on the [Errors](/docs/sdk/go/latest/Errors) page. Errors are returned as values and implement the standard `error` interface; use `errors.As` to match them.

## Versioning

Go SDK releases use semantic versioning. The library is currently in a beta (0.X) state. Breaking changes may be included in `0.X.0` releases.
