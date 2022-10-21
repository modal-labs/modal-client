# Copyright Modal Labs 2022
# Copyright (c) Modal Labs 2022

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
def check_copyright(ctx, fix=False):
    invalid_files = []
    d = str(Path(__file__).parent)
    for root, dirs, files in os.walk(d):
        fns = [os.path.join(root, fn) for fn in files if fn.endswith(".py")]
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
def update_build_number(ctx, build_number):
    with open("modal_version/_version_generated.py", "w") as f:
        f.write(f"{copyright_header_full}\nbuild_number = {build_number}\n")
