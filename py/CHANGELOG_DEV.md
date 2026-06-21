# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**
- `Sandbox.create_connect_token` now accepts a `port` keyword argument (default `8080`) that specifies the container port that requests are routed to when using the token. Port can be between 1 and 65535.
- Added proxy support for the Modal Python client. All Modal API access now respect standard proxy environment variables (`HTTPS_PROXY`, `ALL_PROXY`). HTTP CONNECT and SOCKS4/5 proxies are supported. Support requires the `api-proxy-support` extra dependencies, e.g. `pip install 'modal[api-proxy-support]'`. Set `MODAL_DISABLE_API_PROXY=1` to disable proxy support entirely.
- Added `Sandbox._experimental_set_outbound_network_policy(...)` to update a running Sandbox's outbound network access. Accepts `outbound_cidr_allowlist` and `outbound_domain_allowlist` arguments matching `Sandbox.create`.
- Added `Workspace.billing` and `Environment.billing` which both expose a `.report()` method - this returns comprehensive cost data as a list of `BillingReportItem` dataclasses. `*.report()` has the same parameters as `workspace_billing_report`. `EnvironmentBillingManager.report()` returns data that is specifically scoped to the calling environment.
- Added `modal environment billing report` which wraps `Environment.billing.report()`.
- Deprecated `modal.billing.workspace_billing_report()`
