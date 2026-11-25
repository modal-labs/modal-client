# SDK development changelogs

User-facing updates in the current development versions of our SDKs are tracked here.

Draft releases notes should be added as part of the PR introducing the change. During a release, the notes are moved to the language-specific `CHANGELOG.md` files and edited for publication.

## Python

- NetworkFileSystem has been retired. Removed `modal.NetworkFileSystem`, the `network_file_systems` parameter of `App.function()`, `App.cls()`, `Image.run_function()` and `Sandbox.create()`, and the `modal nfs` CLI.
- The client requires Python >=3.11; Python 3.10 and Images using Python 3.10 are not supported. The minimum supported protobuf version is 4.21.3.
- Added `modal.Image.objects.delete(image_id)` to delete an Image or a Sandbox filesystem or directory snapshot by ID. It replaces the deprecated `modal.experimental.image_delete()`.
- Added a `platform` parameter (`"linux/amd64"` or `"linux/arm64"`) to `Image.from_registry()`, `Image.from_dockerfile()`, `Image.debian_slim()`, `Image.micromamba()` and `Image.from_scratch()`, and a `--platform` option to `modal shell`, `modal image pull` and `modal image build`. Without `platform`, `Image.debian_slim()`, `Image.micromamba()` and `Image.from_scratch()` stay `linux/amd64`. Layers added to an Image, and the Functions and Sandboxes that use it, run on the platform of the Image.
- Added `Workspace.audit_logs.fetch()` and `Workspace.audit_logs.count()` to read and count workspace audit events from the SDK. Counting requires an explicit `since` datetime. The `modal workspace audit-logs` CLI streams full events as newline-delimited JSON.
- Added `modal workspace audit-logs --csv` to export workspace audit events as CSV.
- `Sandbox.wait_until_ready()` now works without a readiness probe, returning when the Sandbox has started.
- `Sandbox.wait()` now raises `ConflictError` with the startup error when a Sandbox fails to initialize.
- Fixed a bug where `.map()`, `.starmap()`, and `.for_each()` could hang indefinitely when called with no inputs.
- Fixed a bug where `.map()`, `.starmap()`, and `.for_each()` could retry an input indefinitely after repeated container crashes. An input now fails with `InternalFailure` after eight such failures, and these failures no longer use up the Function's `Retries` budget.
- Instances of `modal.parameter()` classes that are created before the App is running now validate and encode their parameters the same way as instances created afterwards. Positional arguments and values that don't match the declared parameter type now raise an error when the instance is created, instead of being passed through to the container.
- Fixed classes whose `modal.parameter()` declarations are all inherited from a base class: they previously failed with a `TypeError` when the instance was constructed in the container. Inherited parameters are now type-checked when the class is decorated, and `@app.server` classes reject inherited parameters.
- Updated `modal workspace member-tokens delete` and `Workspace.member_tokens.delete()` such that they can now take in multiple token IDs for batch deletion.
- Added `Workspace.service_users.rotate()` to easily rotate Service User credentials.
- Deprecated `ServiceUserInfo.token_id` and added `ServiceUserInfo.tokens`.
- Added `modal.OutboundPolicy` and `modal.OutboundRoute`, set with `Sandbox.create(outbound_policy=...)` and replaced on a running Sandbox with `Sandbox.set_outbound_policy()`. An outbound policy is an ordered list of routes that decides, by IP range or domain, which outbound traffic from a Sandbox is allowed or blocked. HTTPS routes can also set headers on matching requests from a Secret's values without the Secret entering the Sandbox; header replacements are in public beta.
- Deprecated the `outbound_cidr_allowlist` and `outbound_domain_allowlist` parameters of `Sandbox.create()` and `Sandbox._experimental_set_outbound_network_policy()` in favor of `outbound_policy` and `Sandbox.set_outbound_policy()`. They will be removed in a future release.
- Deprecated `modal.experimental.OutboundPolicy`: `Sandbox.create(_experimental_outbound_policy=...)` and `Sandbox._experimental_update_outbound_policy()` will be removed in modal 1.7.0. Use `outbound_policy` with `modal.OutboundRoute.https(..., header_replacements=...)` routes instead.

## JS

- `client.images.fromRegistry()` now selects the registry auth type (AWS ECR, GCP Artifact Registry, or username and password) from the keys of its Secret, as `Image.from_registry()` does in Python. A Secret with the keys for none or more than one of these makes the build fail with `InvalidError`.
- Added a `platform` option (`"linux/amd64"` or `"linux/arm64"`) to `client.images.fromRegistry()`, which now also takes `{ secret, platform }` as its second argument; a `Secret` as the second argument still works.
- Added `OutboundPolicy` and `OutboundRoute`, set with the `outboundPolicy` option of `client.sandboxes.create()` and replaced on a running Sandbox with `Sandbox.setOutboundPolicy()`. An outbound policy is an ordered list of routes that decides, by IP range or domain, which outbound traffic from a Sandbox is allowed or blocked. HTTPS routes can also set headers on matching requests from a Secret's values without the Secret entering the Sandbox; header replacements are in public beta.
- Deprecated the `outboundCidrAllowlist` and `outboundDomainAllowlist` options of `client.sandboxes.create()` and `Sandbox.updateNetworkPolicy()` in favor of `outboundPolicy` and `Sandbox.setOutboundPolicy()`. They log a warning and will be removed in a future release.
- Deprecated `ExperimentalOutboundPolicy`: the `experimentalOutboundPolicy` option of `client.sandboxes.create()` and `Sandbox.experimentalUpdateOutboundPolicy()` log a warning and will be removed in version 0.12.0. Use `outboundPolicy` with `OutboundRoute.https({ domains, headerReplacements })` routes instead.
- `Function.remote()` on a Function that isn't on the input plane now throws `InternalFailure` when Modal reports an internal failure. It used to retry through an RPC the server no longer accepts, which ended in a gRPC error instead.

## Go

- `Images.FromRegistry()` now selects the registry auth type (AWS ECR, GCP Artifact Registry, or username and password) from the keys of `ImageFromRegistryParams.Secret`, as `Image.from_registry()` does in Python. A Secret with the keys for none or more than one of these makes `Build` fail with `InvalidError`.
- Added a `Platform` field (`"linux/amd64"` or `"linux/arm64"`) to `ImageFromRegistryParams`.
- Added `OutboundPolicy` and its route types, set with the `OutboundPolicy` field of `SandboxCreateParams` and replaced on a running Sandbox with `Sandbox.SetOutboundPolicy()`. An outbound policy is an ordered list of routes that decides, by IP range or domain, which outbound traffic from a Sandbox is allowed or blocked. HTTPS routes can also set headers on matching requests from a Secret's values without the Secret entering the Sandbox; header replacements are in public beta.
- Deprecated the `OutboundCIDRAllowlist` and `OutboundDomainAllowlist` fields of `SandboxCreateParams` and `Sandbox.UpdateNetworkPolicy()` in favor of `OutboundPolicy` and `Sandbox.SetOutboundPolicy()`. They log a warning and will be removed in a future release.
- Deprecated `ExperimentalOutboundPolicy`: the `ExperimentalOutboundPolicy` field of `SandboxCreateParams` and `Sandbox.ExperimentalUpdateOutboundPolicy()` log a warning and will be removed in version 0.12.0. Use `OutboundPolicy` with `OutboundRouteHTTPS` routes that set `HeaderReplacements` instead.
- `Function.Remote()` on a Function that isn't on the input plane now returns `InternalFailure` when Modal reports an internal failure. It used to retry through an RPC the server no longer accepts, which ended in a gRPC error instead.
