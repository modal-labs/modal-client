# Copyright Modal Labs 2022
# Copyright (c) Modal Labs 2022

import inspect

if not hasattr(inspect, "getargspec"):
    # Workaround until invoke supports Python 3.11
    # https://github.com/pyinvoke/invoke/issues/833#issuecomment-1293148106
    inspect.getargspec = inspect.getfullargspec  # type: ignore

import datetime
import os
import sys
from pathlib import Path

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
    ctx.run(f"{py_protoc} -I . modal_proto/api.proto")


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
def update_build_number(ctx, new_build_number):
    new_build_number = int(new_build_number)
    from modal_version import build_number as current_build_number

    assert new_build_number > current_build_number
    with open("modal_version/_version_generated.py", "w") as f:
        f.write(f"{copyright_header_full}\nbuild_number = {new_build_number}\n")
