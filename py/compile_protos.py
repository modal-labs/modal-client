# Copyright Modal Labs 2026
# /// script
# requires-python = ">=3.10"
# ///

import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--skip-mypy", action="store_true")

args = parser.parse_args()


mypy_extra = [] if args.skip_mypy else ["--mypy_out=.", "--mypy_grpc_out=."]

grpclib_protoc_command = [
    sys.executable,
    "-m",
    "grpc_tools.protoc",
    "--python_out=.",
    "--grpclib_python_out=.",
    "--grpc_python_out=.",
    *mypy_extra,
    "-I",
    "..",
    "modal_proto/api.proto",
    "modal_proto/task_command_router.proto",
]


# Suppress pkg_resources deprecation warning from grpcio-tools
# See: https://github.com/grpc/grpc/issues/33570
os.environ["PYTHONWARNINGS"] = "ignore:pkg_resources is deprecated"

print(f"{' '.join(grpclib_protoc_command)}")
subprocess.run(grpclib_protoc_command, check=True)


@contextmanager
def python_file_as_executable(path: Path) -> Generator[Path, None, None]:
    if sys.platform == "win32":
        # windows can't just run shebang:ed python files, so we create a .bat file that calls it
        src = f"""@echo off
{sys.executable} {path}
"""
        with NamedTemporaryFile(mode="w", suffix=".bat", encoding="ascii", delete=False) as f:
            f.write(src)

        try:
            yield Path(f.name)
        finally:
            Path(f.name).unlink()
    else:
        yield path


grpc_plugin_pyfile = Path("protoc_plugin/plugin.py")
with python_file_as_executable(grpc_plugin_pyfile) as grpc_plugin_executable:
    plugin_protoc_command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--plugin=protoc-gen-modal-grpclib-python={grpc_plugin_executable}",
        "--modal-grpclib-python_out=.",
        "-I",
        "..",
        "modal_proto/api.proto",
    ]
    print(f"{' '.join(plugin_protoc_command)}")
    subprocess.run(plugin_protoc_command, check=True)
