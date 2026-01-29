# Changelog for Unreleased User-facing Updates

**When releasing, move these changelog items to `CHANGELOG.md`.**

* Modal objects now have a `.get_dashboard_url()` method.
* There is a new `modal dashboard` CLI and `modal app dashboard` and `modal volume dashboard` CLI subcommands.
* Fixed an issue where Cls.with_options()/with_concurrency()/with_batching() multiple times on the same class could sometimes use stale argument values

