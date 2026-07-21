# Modal SDKs
[![License](https://img.shields.io/badge/license-apache_2.0-darkviolet.svg)](https://github.com/modal-labs/modal-client/blob/master/LICENSE)
[![Slack](https://img.shields.io/badge/slack-join-blue.svg?logo=slack)](https://modal.com/slack)

This repository contains the source code for [Modal's](https://modal.com) official SDKs.

See the [online documentation](https://modal.com/docs) for a detailed [user guide](https://modal.com/docs/guide) and many [example applications](https://modal.com/docs/examples).

For usage questions or other support, please reach out on the [Modal Slack](https://modal.com/slack).

## Python SDK

[![PyPI Version](https://img.shields.io/pypi/v/modal.svg)](https://pypi.org/project/modal/)

The [Modal](https://modal.com/) Python SDK allows you to deploy high-performance serverless applications and programmatically interact with the Modal platform.

See the [Python reference](https://modal.com/docs/sdk/py) for details on usage.

Install the package with `uv` or `pip`:

```bash
uv pip install modal
```

The Python package includes the `modal` CLI. The CLI can also be installed as a standalone tool via [uvx.sh](https://uvx.sh):

```bash
curl -LsSf uvx.sh/modal/install.sh | sh
```

## JavaScript/TypeScript and Go SDKs

[![JS npm Version](https://img.shields.io/npm/v/modal.svg)](https://www.npmjs.org/package/modal)
[![Go Reference Documentation](https://pkg.go.dev/badge/github.com/modal-labs/modal-client/go)](https://pkg.go.dev/github.com/modal-labs/modal-client/go)

The JS and Go SDKs allow you to use Modal Sandboxes, invoke deployed Modal Functions, and interact with some Modal platform resources.

For more details, see the reference documentation:

- [JavaScript / TypeScript](https://modal.com/docs/sdk/js)
- [Go](https://modal.com/docs/sdk/go)

## Skills

Modal distributes an [official skill](./py/modal/skills/modal/SKILL.md) to help coding agents use its latest features. The skill can be installed and maintained via the `modal` CLI:

```bash
modal skills install
```

When the skill is installed via the `modal` CLI, the skill references will be populated with version-aligned documentation.

The skill can also be managed via `npx skills`:

```bash
npx skills add modal-labs/modal-client
```

Note that installation via `npx skills` will not include versioning or reference documentation.
