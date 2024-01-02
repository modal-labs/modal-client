# Copyright Modal Labs 2022
# Copyright (c) Modal Labs 2022

import inspect
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
    ctx.run("mypy .", pty=True)


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
    with open("modal_version/_version_generated.py", "w") as f:
        f.write(
            f"""\
{copyright_header_full}
build_number = {new_build_number}
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
        "modal.proxy",
        "modal.queue",
        "modal.sandbox",
        "modal.secret",
        "modal.stub",
        "modal.volume",
    ]
    subprocess.check_call(["python", "-m", "synchronicity.type_stubs", *modules])
