package modal

// sdkVersion is the checked-in version of the Modal Go SDK.
//
// Keep this in sync with the Go release tags (go/vX.Y.Z) and the JS SDK
// version in client/js/package.json — the release tooling derives both the
// js/vX.Y.Z and go/vX.Y.Z tags from js/package.json. The `inv lint-versions`
// linter enforces that this constant matches js/package.json.
const sdkVersion = "0.9.0"
