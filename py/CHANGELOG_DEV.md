# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**
- `Sandbox.create_connect_token` now accepts a `port` keyword argument (default `8080`) that specifies the container port that requests are routed to when using the token. Port can be between 1 and 65535.
- Added `Sandbox._experimental_set_outbound_network_policy(...)` to update a running Sandbox's outbound network access. Accepts `outbound_cidr_allowlist` and `outbound_domain_allowlist` arguments matching `Sandbox.create`.
- Added `Workspace.billing` and `Environment.billing` which both expose a `.report()` method - this returns comprehensive cost data as a list of `BillingReportItem` dataclasses. `*.report()` has the same parameters as `workspace_billing_report`. `EnvironmentBillingManager.report()` returns data that is specifically scoped to the calling environment.
- Added `modal environment billing report` which wraps `Environment.billing.report()`.
- Deprecated `modal.billing.workspace_billing_report()`
