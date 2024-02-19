# Copyright Modal Labs 2022
# Copyright (c) Modal Labs 2022

import inspect
import re
import subprocess

if not hasattr(inspect, "getargspec"):
    # Workaround until invoke supports Python 3.11
    # https://github.com/pyinvoke/invoke/issues/833#issuecomment-1293148106
    inspect.getargspec = inspect.getfullargspec  # type: ignore

import datetime
import os
import sys
from pathlib import Path
from typing import Optional

import requests
from invoke import task

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
def lint(ctx):
    ctx.run("ruff .", pty=True)


@task
def mypy(ctx):
    mypy_allowlist = [
        "modal/functions.py",
    ]

    ctx.run("mypy .", pty=True)
    ctx.run(f"mypy {' '.join(mypy_allowlist)} --follow-imports=skip", pty=True)


@task
def check_copyright(ctx, fix=False):
    invalid_files = []
    d = str(Path(__file__).parent)
    for root, dirs, files in os.walk(d):
        # jupytext notebook formatted .py files can't be detected as notebooks if we put a copyright comment at the top
        fns = [
            os.path.join(root, fn)
            for fn in files
            if fn.endswith(".py") and not fn.endswith(".notebook.py") and "/site-packages/" not in root
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
        "modal.s3mount",
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
