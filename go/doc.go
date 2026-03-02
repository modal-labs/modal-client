// Package modal is a lightweight, idiomatic Go SDK for Modal.com.
//
// It mirrors the core feature-set of Modal’s Python SDK while feeling
// natural in Go:
//
//   - Spin up Sandboxes — fast, secure, ephemeral VMs for running code.
//   - Invoke Modal Functions and manage their inputs / outputs.
//   - Read, write, and list files in Modal Volumes.
//   - Create or inspect containers, streams, and logs.
//
// **What it does not do:** deploying Modal Functions. Deployment is still
// handled in Python; this package is for calling and orchestrating them
// from other projects.
//
// # Configuration
//
// The config file path can be customized via `MODAL_CONFIG_PATH` (defaults to `~/.modal.toml`).
//
// ## Authentication
//
// At runtime the SDK resolves credentials in this order:
//
//  1. Environment variables
//     MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, MODAL_ENVIRONMENT (optional)
//  2. A profile explicitly requested via `MODAL_PROFILE`
//  3. A profile marked `active = true` in `~/.modal.toml`
//
// ## Logging
//
// The SDK logging level can be controlled in multiple ways (in order of precedence):
//
//  1. `MODAL_LOGLEVEL` environment variable
//  2. `loglevel` field in the active profile in `~/.modal.toml`
//  3. Defaults to WARN
//
// Supported values are DEBUG, INFO, WARN, and ERROR (case-insensitive).
//
// Logs are written to stderr.
//
// For additional examples and language-parity tests, see
// https://github.com/modal-labs/modal-client/tree/main/go.
package modal
