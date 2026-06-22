# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**
- `Image.from_id(...)` references can now `.publish()` an image by ID without first calling `.build()` on the Image.
- Improves `Sandbox.exec` latency when passing in `Secret.from_dict`, `Secret.from_dotenv`, and `Secret.from_local_environ`.
- `Sandbox.create_connect_token` now accepts a `port` keyword argument (default `8080`) that specifies the container port that requests are routed to when using the token. Port can be between 1 and 65535.
- Added proxy support for the Modal Python client. All Modal API access now respect standard proxy environment variables (`HTTPS_PROXY`, `ALL_PROXY`). HTTP CONNECT and SOCKS4/5 proxies are supported. Support requires the `api-proxy-support` extra dependencies, e.g. `pip install 'modal[api-proxy-support]'`. Set `MODAL_DISABLE_API_PROXY=1` to disable proxy support entirely.
- Added `Sandbox._experimental_set_outbound_network_policy(...)` to update a running Sandbox's outbound network access. Accepts `outbound_cidr_allowlist` and `outbound_domain_allowlist` arguments matching `Sandbox.create`.
- Added `Workspace.billing` and `Environment.billing` which both expose a `.report()` method - this returns comprehensive cost data as a list of `BillingReportItem` dataclasses. `*.report()` has the same parameters as `workspace_billing_report`. `EnvironmentBillingManager.report()` returns data that is specifically scoped to the calling environment.
- Added `modal environment billing report` which wraps `Environment.billing.report()`.
- Added `Workspace.proxy_tokens` for managing proxy tokens. It exposes `.create()` (returning a `TokenData` with `token_id` and `token_secret`), `.list(environment_name=None)` (returning `ProxyTokenInfo` dataclasses, optionally filtered to a given environment), `.allow(proxy_token_id, environment_name)` and `.revoke(proxy_token_id, environment_name)` (to manage environment associations), and `.delete(proxy_token_id)`.
- Added `modal workspace` CLI with `modal workspace proxy-tokens` commands (`create`, `list` [`--environment`], `allow`, `revoke`, `delete`) for managing the current Workspace's proxy tokens, plus `modal workspace members list`. `create` prints the new token's `Modal-Key` and `Modal-Secret` request headers (pass `--json` to emit them as JSON).
- Deprecated `modal.billing.workspace_billing_report()`
- Added `--strategy` option to `modal app rollback`. As with `modal app rollover`, there are two strategies for switching between deployments:
  - `--strategy=rolling` (the default) will smoothly migrate traffic from old containers to new containers
  - `--strategy=recreate` will terminate all running containers so that any subsequent inputs will go to new containers
- Renamed the `region` parameter of the `@app.server()` decorator to `compute_region`.
