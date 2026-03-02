# Modal SDKs

## Documentation

See the [online documentation](https://modal.com/docs/guide) for many
[example applications](https://modal.com/docs/examples),
a [user guide](https://modal.com/docs/guide), and the detailed
[API reference](https://modal.com/docs/reference).

## Python SDK

[![PyPI Version](https://img.shields.io/pypi/v/modal.svg)](https://pypi.org/project/modal/)
[![License](https://img.shields.io/badge/license-apache_2.0-darkviolet.svg)](https://github.com/modal-labs/modal-client/blob/master/LICENSE)
[![Tests](https://github.com/modal-labs/modal-client/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/modal-labs/modal-client/actions/workflows/ci-cd.yml)
[![Slack](https://img.shields.io/badge/slack-join-blue.svg?logo=slack)](https://modal.com/slack)

The [Modal](https://modal.com/) Python SDK provides convenient, on-demand
access to serverless cloud compute from Python scripts on your local computer.

**This library requires Python 3.10 – 3.14.**

### Installation

Install the package with `uv` or `pip`:

```bash
uv pip install modal
```

You can create a Modal account (or link your existing one) directly on the
command line:

```bash
python3 -m modal setup
```

## JavaScript/TypeScript and Go SDKs

[![JS Reference Documentation](https://img.shields.io/static/v1?message=reference&logo=javascript&labelColor=5c5c5c&color=1182c3&logoColor=white&label=%20)](https://modal-labs.github.io/libmodal/)
[![JS npm Version](https://img.shields.io/npm/v/modal.svg)](https://www.npmjs.org/package/modal)
[![JS npm Downloads](https://img.shields.io/npm/dm/modal.svg)](https://www.npmjs.com/package/modal)
[![Go Reference Documentation](https://pkg.go.dev/badge/github.com/modal-labs/modal-client/go)](https://pkg.go.dev/github.com/modal-labs/modal-client/go)

This repo hosts the [Modal](https://modal.com) SDKs for JavaScript/TypeScript and Go. They provide convenient, on-demand access to serverless cloud compute on Modal from JS/TS and golang projects. Use it to safely run arbitrary code in Modal Sandboxes, call Modal Functions, and interact with Modal resources.

For more details, documentation and installation instructions, see the README for each SDK:
- **[JavaScript / TypeScript](./js/README.md)**
- **[Go](./go/README.md)**


## Community SDKs

There are also open-source Modal libraries built and maintained by our community. These projects are not officially supported by Modal and we thus can't vouch for them, but feel free to explore and contribute.

- Ruby: [anthonycorletti/modal-rb](https://github.com/anthonycorletti/modal-rb)


## Support

For usage questions and other support, please reach out on the
[Modal Slack](https://modal.com/slack).
