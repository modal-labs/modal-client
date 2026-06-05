# Contributing to the Modal JS and Go SDKs

## Development principles

**We aim to maintain identical behavior across languages:**

- When merging a feature or change into `main`, update it for all languages
  simultaneously, with tests.
- Code structure should be similar between language folders.
- Use a common set of gRPC primitives (retries, deadlines) and exceptions.
- Complex types like streams must behave as similarly as possible.

Notable differences:

- Timeouts use milliseconds in JavaScript/TypeScript and `time.Duration` in Go.

## Tests

Tests are run against Modal cloud infrastructure, and you need to be
authenticated with Modal to run them. See the [`test-support/`](./test-support)
folder for details.

## modal-js development

Navigate to the `client/js` folder and install dependencies:

```bash
cd client/js
npm install
```

Run a script:

```bash
node --import tsx examples/sandbox.ts
```

### JS naming conventions

- Parameters should always include explicit unit suffixes to make the API more
  self-documenting and prevent confusion about units:
  - durations should be suffixed with `Ms`, e.g. `timeoutMs` instead of
    `timeout`
  - memory should be suffixed with `MiB`, e.g. `memoryMiB` instead of `memory`

### gRPC support

We're using `nice-grpc` because the `@grpc/grpc-js` library doesn't support
promises and is difficult to customize with types.

This gRPC library depends on the `protobuf-ts` package, which is not compatible
with tree shaking because `ModalClientDefinition` transitively references every
type. However, since `modal-js` is a server-side package, having a larger
bundled library is not a huge issue.

## modal-go development

Navigate to the `client/go` folder and you should be all set to run an example:

```bash
cd client/go
go run examples/sandbox/main.go
```

Whenever you need a new version of the protobufs, regenerate the protobuf files
with:

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
scripts/gen-proto.sh
```

We check the generated protobuf files into Git so that the package can be
installed with `go get`.

### Go naming conventions

- Parameters should always include explicit unit suffixes to make the API more
  self-documenting and prevent confusion about units:
  - durations should NOT be suffixed, since they have type `time.Duration`
  - memory should be suffixed with `MiB`, e.g. `memoryMiB` instead of `memory`

### Testing

We use [goleak](https://github.com/uber-go/goleak) for goroutine leak detection.
To identify which specific test is leaking, use
[goleaks recommended method to find them](https://github.com/uber-go/goleak/?tab=readme-ov-file#determine-source-of-package-leaks),
then use `go test -run` to run individual tests.

## Releasing the Go/JS SDK

1. Navigate to the client directory and run `inv update-version-go-js`:

```bash
cd client
# For major release
inv update-version-go-js --patch major
# For minor release
inv update-version-go-js --patch minor
# For patch release
inv update-version-go-js --patch patch
```

2. You can change the wording or order of items in `client/CHANGELOG_GO_JS.md`
3. Open PR with your changes.
