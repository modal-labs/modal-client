# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

- We recommend calling `Sandbox.detach` after you are done interacting with a sandbox. `Sandbox.detach` disconnects your client from the sandbox and cleans up resources associated with the connection. After calling `detach`, any operation using the Sandbox object is not guaranteed to work. If you want to continue interacting with a running sandbox, use `Sandbox.from_id` to get a new Sandbox object.
- `Sandbox.terminate` now accepts a `wait` parameter. If `wait=True`, `terminate` will wait for the sandbox to terminate and return the exit code.
