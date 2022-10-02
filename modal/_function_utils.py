import inspect
import os
import site
import sys
import sysconfig
import typing
from pathlib import Path
from typing import Dict, Union

import cloudpickle

from modal_proto import api_pb2

from .config import config, logger
from .mount import _Mount

ROOT_DIR = "/root"

# Expand symlinks in paths (homebrew Python paths are all symlinks).
SYS_PREFIXES = set(
    os.path.realpath(p)
    for p in (
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        *sysconfig.get_paths().values(),
        *site.getsitepackages(),
        site.getusersitepackages(),
    )
)


def package_mount_condition(filename):
    if filename.startswith(sys.prefix):
        return False

    return os.path.splitext(filename)[1] in [".py"]


def _is_modal_path(remote_path: Union[str, Path]):
    parts = os.path.split(remote_path)
    is_modal_path = parts[:2] in [
        ("/root", "modal"),
        ("/root", "modal_proto"),
        ("/root", "modal_utils"),
        ("/root", "modal_version"),
    ]
    return is_modal_path


def filter_safe_mounts(mounts: typing.Dict[str, _Mount]):
    # exclude mounts that would overwrite Modal
    return {local_dir: mount for local_dir, mount in mounts.items() if not _is_modal_path(mount._remote_dir)}


class FunctionInfo:
    """Class the helps us extracting a bunch of information about a function."""

    # TODO: we should have a bunch of unit tests for this
    # TODO: if the function is declared in a local scope, this function still "works": we should throw an exception
    def __init__(self, f, serialized=False):
        self.function_name = f.__qualname__
        self.function_serialized = None
        self.signature = inspect.signature(f)
        module = inspect.getmodule(f)

        if getattr(module, "__package__", None) and not serialized:
            # This is a "real" module, eg. examples.logs.f
            # Get the package path
            # Note: __import__ always returns the top-level package.
            module_file = module.__file__
            package_paths = __import__(module.__package__).__path__
            # There might be multiple package paths in some weird cases
            base_dirs = [
                base_dir for base_dir in package_paths if os.path.commonpath((base_dir, module_file)) == base_dir
            ]

            if len(base_dirs) != 1:
                logger.info(f"Module files: {module_file}")
                logger.info(f"Package paths: {package_paths}")
                logger.info(f"Base dirs: {base_dirs}")
                raise Exception("Wasn't able to find the package directory!")
            (self.base_dir,) = base_dirs
            self.module_name = module.__spec__.name
            self.remote_dir = os.path.join(ROOT_DIR, module.__package__.split(".")[0])
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
            self.is_package = True
        elif hasattr(module, "__file__") and not serialized:
            # This generally covers the case where it's invoked with
            # python foo/bar/baz.py
            self.file = inspect.getfile(f)
            self.module_name = inspect.getmodulename(self.file)
            self.base_dir = os.path.dirname(self.file)
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
            self.is_package = False
            self.is_file = True
        else:
            # Use cloudpickle. Used when working w/ Jupyter notebooks.
            self.function_serialized = cloudpickle.dumps(f)
            logger.debug(f"Serializing {f.__qualname__}, size is {len(self.function_serialized)}")
            self.module_name = None
            self.base_dir = os.path.abspath("")  # get current dir
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_SERIALIZED
            self.is_package = False
            self.is_file = False

    def get_mounts(self) -> Dict[str, _Mount]:
        if self.is_package:
            mounts = {
                self.base_dir: _Mount(
                    local_dir=self.base_dir,
                    remote_dir=self.remote_dir,
                    recursive=True,
                    condition=package_mount_condition,
                )
            }
        elif self.is_file:
            mounts = {
                self.file: _Mount(
                    local_file=self.file,
                    remote_dir=ROOT_DIR,
                )
            }
        else:
            return {}

        if not config.get("automount"):
            return filter_safe_mounts(mounts)

        # Auto-mount local modules that have been imported in global scope.
        # Note: sys.modules may change during the iteration
        modules = []
        skip_prefixes = set()
        for name, module in sorted(sys.modules.items(), key=lambda kv: len(kv[0])):
            parent = name.rsplit(".")[0]
            if parent and parent in skip_prefixes:
                skip_prefixes.add(name)
                continue
            skip_prefixes.add(name)
            modules.append(module)

        for m in modules:
            if getattr(m, "__package__", None):
                package_path = __import__(m.__package__).__path__
                for raw_path in package_path:
                    path = os.path.realpath(raw_path)

                    if (
                        path in mounts
                        or any(raw_path.startswith(p) for p in SYS_PREFIXES)
                        or any(path.startswith(p) for p in SYS_PREFIXES)
                        or not os.path.exists(path)
                    ):
                        continue

                    mounts[path] = _Mount(
                        local_dir=path,
                        remote_dir=os.path.join(ROOT_DIR, *m.__name__.split(".")),
                        condition=package_mount_condition,
                        recursive=True,
                    )
            elif getattr(m, "__file__", None):
                path = os.path.abspath(os.path.realpath(m.__file__))

                if (
                    path in mounts
                    or any(m.__file__.startswith(p) for p in SYS_PREFIXES)
                    or any(path.startswith(p) for p in SYS_PREFIXES)
                    or not os.path.exists(path)
                ):
                    continue
                relpath = os.path.relpath(os.path.dirname(path), self.base_dir)
                mounts[path] = _Mount(
                    local_file=path,
                    remote_dir=os.path.join(ROOT_DIR, relpath),
                )
        return filter_safe_mounts(mounts)

    def get_tag(self):
        return self.function_name

    def is_nullary(self):
        return all(param.default is not param.empty for param in self.signature.parameters.values())
