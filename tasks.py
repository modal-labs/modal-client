# Copyright Modal Labs 2022
# Copyright (c) Modal Labs 2022

import ast
import datetime
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import requests
from invoke import task
from rich.console import Console
from rich.table import Table

year = datetime.date.today().year
copyright_header_start = "# Copyright Modal Labs"
copyright_header_full = f"{copyright_header_start} {year}"


@task
def protoc(ctx):
    py_protoc = (
        f"{sys.executable} -m grpc_tools.protoc"
        + " --python_out=. --grpclib_python_out=. --grpc_python_out=. --mypy_out=. --mypy_grpc_out=."
    )
    print(py_protoc)
    ctx.run(f"{py_protoc} -I . " "modal_proto/api.proto " "modal_proto/options.proto ")


@task
def lint(ctx, fix=False):
    ctx.run(f"ruff . {'--fix' if fix else ''}", pty=True)


@task
def type_check(ctx):
    # mypy will not check the *implementation* (.py) for files that also have .pyi type stubs
    ctx.run("mypy . --exclude=playground --exclude=venv311 --exclude=venv38", pty=True)

    # use pyright for checking implementation of those files
    pyright_allowlist = [
        "modal/functions.py",
        "modal_utils/__init__.py",
        "modal_utils/app_utils.py",
        "modal_utils/async_utils.py",
        "modal_utils/grpc_testing.py",
        "modal_utils/http_utils.py",
        "modal_utils/logger.py",
        "modal_utils/package_utils.py",
    ]

    ctx.run(f"pyright {' '.join(pyright_allowlist)}", pty=True)


@task
def check_copyright(ctx, fix=False):
    invalid_files = []
    d = str(Path(__file__).parent)
    for root, dirs, files in os.walk(d):
        fns = [
            os.path.join(root, fn)
            for fn in files
            if (
                fn.endswith(".py")
                # jupytext notebook formatted .py files can't be detected as notebooks if we put a copyright comment at the top
                and not fn.endswith(".notebook.py")
                # vendored code has a different copyright
                and "_vendor" not in root
                # third-party code (i.e., in a local venv) has a different copyright
                and "/site-packages/" not in root
            )
        ]
        for fn in fns:
            first_line = open(fn).readline()
            if not first_line.startswith(copyright_header_start):
                if fix:
                    print(f"Fixing {fn}...")
                    content = copyright_header_full + "\n" + open(fn).read()
                    with open(fn, "w") as g:
                        g.write(content)
                else:
                    invalid_files.append(fn)

    if invalid_files:
        for fn in invalid_files:
            print("Missing copyright:", fn)

        raise Exception(
            f"{len(invalid_files)} are missing copyright headers!" " Please run `inv check-copyright --fix`"
        )


@task
def publish_base_mounts(ctx, no_confirm=False):
    from urllib.parse import urlparse

    from modal import config

    server_url = config.config["server_url"]
    if "localhost" not in urlparse(server_url).netloc and not no_confirm:
        answer = input(f"Modal server URL is '{server_url}' not localhost. Continue operation? [y/N]: ")
        if answer.upper() not in ["Y", "YES"]:
            exit("Aborting task.")
    for mount in ["modal_client_package", "python_standalone"]:
        ctx.run(f"{sys.executable} {Path(__file__).parent}/modal_global_objects/mounts/{mount}.py", pty=True)


@task
def update_build_number(ctx, new_build_number: Optional[int] = None):
    from modal_version import build_number as current_build_number

    new_build_number = int(new_build_number) if new_build_number else current_build_number + 1
    assert new_build_number > current_build_number

    # Add the current Git SHA to the file, so concurrent publish actions of the
    # client package result in merge conflicts.
    git_sha = ctx.run("git rev-parse --short=7 HEAD", hide="out").stdout.rstrip()

    with open("modal_version/_version_generated.py", "w") as f:
        f.write(
            f"""\
{copyright_header_full}

# Note: Reset this value to -1 whenever you make a minor `0.X` release of the client.
build_number = {new_build_number}  # git: {git_sha}
"""
        )


@task
def create_alias_package(ctx):
    from modal_version import __version__

    os.makedirs("alias-package", exist_ok=True)
    with open("alias-package/setup.py", "w") as f:
        f.write(
            f"""\
{copyright_header_full}
from setuptools import setup
setup(version="{__version__}")
"""
        )
    with open("alias-package/setup.cfg", "w") as f:
        f.write(
            f"""\
[metadata]
name = modal-client
author = Modal Labs
author_email = support@modal.com
description = Legacy name for the Modal client
long_description = This is a legacy compatibility package that just requires the `modal` client library.
            In versions before 0.51, the official name of the client library was called `modal-client`.
            We have renamed it to `modal`, but this library is kept updated for compatibility.
long_description_content_type = text/markdown
project_urls =
    Homepage = https://modal.com

[options]
install_requires =
    modal=={__version__}
"""
        )
    with open("alias-package/pyproject.toml", "w") as f:
        f.write(
            """\
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
"""
        )


@task
def type_stubs(ctx):
    # we only generate type stubs for modules that contain synchronicity wrapped types
    # TODO(erikbern): can we automate this list?
    modules = [
        "modal.app",
        "modal.client",
        "modal.cls",
        "modal.dict",
        "modal.environments",
        "modal.functions",
        "modal.image",
        "modal.mount",
        "modal.network_file_system",
        "modal.object",
        "modal.partial_function",
        "modal.proxy",
        "modal.queue",
        "modal.cloud_bucket_mount",
        "modal.sandbox",
        "modal.secret",
        "modal.stub",
        "modal.volume",
    ]
    subprocess.check_call(["python", "-m", "synchronicity.type_stubs", *modules])


@task
def update_changelog(ctx):
    # Parse the most recent commit message for a GitHub PR number
    res = ctx.run("git log --pretty=format:%s -n 1", hide="stdout")
    m = re.search(r"\(#(\d+)\)$", res.stdout)
    if m:
        pull_number = m.group(1)
    else:
        print("Aborting: No PR number in commit message")
        return

    # Get the corresponding PR description via the GitHub API
    url = f"https://api.github.com/repos/modal-labs/modal-client/pulls/{pull_number}"
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers).json()
    pr_description = response.get("body")
    if pr_description is None:
        print("Aborting: No PR description in response from GitHub API")
        return

    # Parse the PR description to get a changelog update
    comment_pattern = r"<!--.+?-->"
    pr_description = re.sub(comment_pattern, "", pr_description, flags=re.DOTALL)

    changelog_pattern = r"## Changelog\s*(.+)$"
    m = re.search(changelog_pattern, pr_description, flags=re.DOTALL)
    if m:
        update = m.group(1).strip()
    else:
        print("Aborting: No changelog section in PR description")
        return
    if not update:
        print("Aborting: Empty changelog in PR description")
        return

    # Read the existing changelog and split after the header so we can prepend new content
    with open("CHANGELOG.md", "r") as fid:
        content = fid.read()
    token_pattern = "<!-- NEW CONTENT GENERATED BELOW. PLEASE PRESERVE THIS COMMENT. -->"
    m = re.search(token_pattern, content)
    if m:
        break_idx = m.span()[1]
        header = content[:break_idx]
        previous_changelog = content[break_idx:]
    else:
        print("Aborting: Could not find token in existing changelog to mark insertion spot")

    # Build the new changelog and write it out
    from modal_version import __version__

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    new_section = f"### {__version__} ({date})\n\n{update}"
    final_content = f"{header}\n\n{new_section}\n\n{previous_changelog}"
    with open("CHANGELOG.md", "w") as fid:
        fid.write(final_content)


@task
def show_deprecations(ctx):
    def get_modal_source_files() -> list[str]:
        source_files: list[str] = []
        for root, _, files in os.walk("modal"):
            for file in files:
                if file.endswith(".py"):
                    source_files.append(os.path.join(root, file))
        return source_files

    class FunctionCallVisitor(ast.NodeVisitor):
        def __init__(self, fname):
            self.fname = fname
            self.deprecations = []
            self.assignments = {}
            self.current_class = None
            self.current_function = None

        def visit_ClassDef(self, node):
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = None

        def visit_FunctionDef(self, node):
            self.current_function = node.name
            self.assignments["__doc__"] = ast.get_docstring(node)
            self.generic_visit(node)
            self.current_function = None
            self.assignments.pop("__doc__", None)

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assignments[target.id] = node.value
            self.generic_visit(node)

        def visit_Attribute(self, node):
            self.assignments[node.attr] = node.value
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "deprecation_warning":
                depr_date = date(*(elt.n for elt in node.args[0].elts))
                function = (
                    f"{self.current_class}.{self.current_function}" if self.current_class else self.current_function
                )
                message = node.args[1]
                if isinstance(message, ast.Name):
                    message = self.assignments.get(message.id, "")
                if isinstance(message, ast.Attribute):
                    message = self.assignments.get(message.attr, "")
                if isinstance(message, ast.Constant):
                    message = message.s
                elif isinstance(message, ast.JoinedStr):
                    message = "".join(v.s for v in message.values if isinstance(v, ast.Constant))
                else:
                    message = str(message)
                message = message.replace("\n", " ")
                if len(message) > (max_length := 80):
                    message = message[:max_length] + "..."
                self.deprecations.append((str(depr_date), f"{self.fname}:{node.lineno}", function, message))

    files = get_modal_source_files()
    deprecations = []
    for fname in files:
        with open(fname) as f:
            tree = ast.parse(f.read())
        visitor = FunctionCallVisitor(fname)
        visitor.visit(tree)
        deprecations.extend(visitor.deprecations)

    console = Console()
    table = Table("Date", "Location", "Function", "Message")
    for row in sorted(deprecations, key=lambda r: r[0]):
        table.add_row(*row)
    console.print(table)
