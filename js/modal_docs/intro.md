---
description: Complete API reference for the Modal JavaScript SDK.
---

# JavaScript SDK Reference

This is the API reference for the [Modal](https://www.npmjs.com/package/modal) JavaScript SDK.

## Installing

Install the library from npm:

```bash
npm install modal
```

The SDK is written in TypeScript and ships complete type definitions. It works in server-side Node.js (Node 22+), Deno, and Bun projects. Both ES Modules and CommonJS formats are bundled, so you can use `import` or `require()`.

## Configuration

A Modal API token / secret pair is required for authentication. Credentials are read from the following sources in descending order of precedence:

- Parameters passed when constructing a new [`ModalClient`](/docs/sdk/js/latest/ModalClient)
- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` environment variables
- `token_id` and `token_secret` fields in the active profile of a `~/.modal.toml` file

API tokens can be created in the [workspace settings](/settings/tokens) on the Modal Dashboard.

Tokens can also be created and managed with the Modal [CLI](/docs/cli/latest). The CLI is packaged with the [Python SDK](/docs/sdk/py/latest), but it can also be installed as a standalone tool:

```bash
curl -LsSf uvx.sh/modal/install.sh | sh
```

## Objects

|                                            |                                                               |
| ------------------------------------------ | ------------------------------------------------------------- |
| [`App`](/docs/sdk/js/latest/App)           | The main unit of deployment for code on Modal                 |
| [`Function`](/docs/sdk/js/latest/Function) | A serverless function backed by an autoscaling container pool |
| [`Image`](/docs/sdk/js/latest/Image)       | An immutable representation of a container filesystem         |
| [`Sandbox`](/docs/sdk/js/latest/Sandbox)   | A process-like interface for restricted execution             |
| [`Secret`](/docs/sdk/js/latest/Secret)     | A secure reference to environment variables                   |
| [`Volume`](/docs/sdk/js/latest/Volume)     | A mutable distributed storage device                          |

## Scope

The JS SDK is more limited in scope than Modal's [Python SDK](/docs/sdk/py/latest). Namely, Python remains the only supported Function runtime, so features for defining Functions are not available in JS. We do aim to support most interactions with remote Modal objects, including Functions, Sandboxes, Volumes, and others. Some in-scope features are not yet implemented; please get in touch if these are important to you.

## Usage

Instances of the Modal Object types are obtained through service methods on the [ModalClient](/docs/sdk/js/latest/ModalClient):

```typescript
import { ModalClient } from "modal";

const modal = new ModalClient();
```

Interaction with instances of the Modal Objects happens via methods on those instances. Most methods have an associated Params type for passing optional arguments.

### Sandbox

The JS SDK aims to support the complete feature-set for Modal [Sandboxes](/docs/guide/sandboxes):

```typescript
const app = await modal.apps.fromName("sandbox-app", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine:3.21");

const sb = await modal.sandboxes.create(app, image, { command: ["cat"] });

await sb.stdin.writeText("Hello from a Sandbox!");
await sb.stdin.close();
console.log(await sb.stdout.readText());

await sb.terminate();
```

Sandboxes accept many additional parameters at creation time for configuring their resources, lifecycle, restrictions, etc.:

```typescript
const secret = await modal.secrets.fromName("github-token");

const sb = await modal.sandboxes.create(app, image, {
  secrets: [secret],
  timeoutMs: 30 * 60 * 1000,
  cpu: 2,
  memoryMiB: 2048,
  outboundDomainAllowlist: ["github.com", "*.githubusercontent.com"],
});
```

### Function

Modal Functions in deployed Apps can be [looked up and invoked](/docs/guide/trigger-deployed-functions) from JS/TS programs:

```typescript
const echo = await modal.functions.fromName("my-app", "echo");

// Call the Function with args
let ret = await echo.remote(["Hello, Modal!"]);
console.log(ret);

// Call the Function with kwargs
ret = await echo.remote([], { s: "Hello, Modal!" });
console.log(ret);
```

All Function invocation methods are available:

```typescript
const functionCall = await echo.spawn(["Hello, Modal!"]);
const ret = await functionCall.get();
console.log(ret);
```

## Errors

Error types are documented on the [Errors](/docs/sdk/js/latest/Errors) page. Methods throw typed error classes; use `instanceof` to match them.

## Versioning

JS SDK releases use semantic versioning. The library is currently in a beta (0.X) state. Breaking changes may be included in `0.X.0` releases.
