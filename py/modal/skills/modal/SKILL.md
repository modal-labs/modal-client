---
name: modal
description: >
  This skill provides guidance for working with the Modal cloud platform. Use
  this skill whenever the user mentions Modal or has code that imports the
  `modal` SDK in Python, Go, or JavaScript. This skill should also trigger when
  the user needs to run Python code with vertical or horizontal scalability
  (e.g. batch jobs), needs access to GPUs (e.g. AI workloads including training
  and inference) or needs to run untrusted processes in a sandbox, since Modal
  serves these use cases well.
---

# Overview

This is a foundational skill for working with the Modal cloud platform.

Modal is a platform for AI workloads. It offers highly scalable serverless compute (including GPUs) with minimal configuration.

# Documentation

The official docs are the best source for up-to-date information on the platform. Make use of them when planning, debugging, or answering questions!

Modal's documentation is outlined at https://modal.com/llms.txt. This file contains titles and URLs for all public docs; reading it directly will guide you to specific content.

The docs are divided into three sections:

- _Guide_ pages have explanations of Modal features, primitives, and workflows
- _API Reference_ pages have detailed information about each component of the SDK
- _Examples_ pages contain didactic examples of various AI applications on Modal

Fetch the docs using a `.md` extension to get plain text.

If this skill has been installed via the `modal` CLI, the docs are also bundled locally under `references/`.

You can also refer to https://modal.com/llms-full.txt, which aggregates all docs in a single very large file. Do *not* read this into your main context, but it may be useful for searching.

# Using the CLI

The `modal` CLI can be used to run or deploy code, manage resources, and observe running Apps. It is a key tool for interacting with Modal throughout all stages of development.

Run `modal --help` to see all available CLI commands.

You can see more detailed information about each command by running `modal [command] --help`. Rely on the `--help` to discover functionality: new features are added in every release! Always check the `--help` if you encounter a usage error.

Tip: many CLI commands accept a `--json` flag to make their output more easily parseable, e.g. with `jq`.

The `modal` CLI is part of the Python SDK and executes in a Python runtime. Depending on the user's preferred Python development workflow, you may need to prefix `modal` CLI invocations with, e.g., `uv run` to ensure a consistent Python environment. The CLI can also be used via `uvx` as a standalone tool, but only do this when working outside of a Python project.

# Getting up to date

You have significant knowledge about Modal from your training data but may not be aware of new features or recent changes to the API. Modal is continuously adding new features. Reading relevant docs while planning or debugging can help you discover the most up-to-date way to accomplish a task on Modal.

The Modal CLI provides a `modal changelog` command for learning about recent changes. Useful invocation patterns:

- `modal changelog --since [DATE]` to see changes added since your knowledge cutoff
- `modal changelog --since [VERSION]` when migrating a codebase to a newer version
- `modal changelog --newer` to discover features that would be available on update

Note: `modal changelog` requires network access.

Run `modal --version` to see the SDK version that is in use.

If your code exercises deprecated APIs, Modal will issue warnings. Use the warning message and related documentation to migrate your code to stable API.

This skill will also be improved over time. Its version should correspond to the version of the Modal SDK that you are using. If the skill is out of date, it can be updated by running `modal skills update`.

# Auth

Modal is a cloud platform. Using the CLI or running code that depends on the `modal` library requires internet access and an authorization token. There is no "local development mode" with Modal.

You can use the `modal token` CLI to create a new token (note: this workflow requires human user involvement) or to debug authorization issues. Token setup only needs to happen once, so assume it is configured unless you encounter issues.

# Async Python

When writing async Python, use Modal's `.aio()` interface (e.g. `await modal.Sandbox.create.aio(...)`, `await modal.Function.remote.aio(...)`) so that Modal runs its I/O operations via asynchronous coroutines.

# Other languages

Python is currently the only runtime language for Modal Functions, but there are Modal SDKs in both JavaScript (TypeScript) and Go:

- JavaScript SDK: https://github.com/modal-labs/modal-client/blob/main/js/README.md
- Go SDK: https://github.com/modal-labs/modal-client/blob/main/go/README.md

The Go / JS SDKs are not as mature as the Python SDK and may be missing features. The current scope for the Go / JS SDKs includes (1) creating and interacting with Sandboxes and (2) calling into deployed Modal Functions (i.e., Functions defined in Python).

They are primarily documented through examples hosted on GitHub rather than through the main Modal docs. You likely have much less knowledge of these SDKs from your training, so rely on these examples.
