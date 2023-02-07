# Copyright Modal Labs 2022
import inspect
import os
import site
import sys
import sysconfig
import typing
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Optional, Type

from modal_proto import api_pb2

from ._serialization import serialize
from .config import config, logger
from .exception import InvalidError
from .mount import _Mount

ROOT_DIR = PurePosixPath("/root")

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


class FunctionInfoType(Enum):
    PACKAGE = "package"
    FILE = "file"
    SERIALIZED = "serialized"
    NOTEBOOK = "notebook"


class LocalFunctionError(InvalidError):
    """Raised if a function declared in a non-global scope is used in an impermissible way"""

    pass


def package_mount_condition(filename):
    if filename.startswith(sys.prefix):
        return False

    return os.path.splitext(filename)[1] in [".py"]


def _is_modal_path(remote_path: PurePosixPath):
    path_prefix = remote_path.parts[:3]
    is_modal_path = path_prefix in [
        ("/", "root", "modal"),
        ("/", "root", "modal_proto"),
        ("/", "root", "modal_utils"),
        ("/", "root", "modal_version"),
    ]
    return is_modal_path


def filter_safe_mounts(mounts: typing.Dict[str, _Mount]):
    # exclude mounts that would overwrite Modal
    safe_mounts = {}
    for local_dir, mount in mounts.items():
        for entry in mount._entries:
            if _is_modal_path(entry.remote_path):
                break
        else:
            safe_mounts[local_dir] = mount
    return safe_mounts


def is_global_function(function_qual_name):
    return "<locals>" not in function_qual_name.split(".")


class FunctionInfo:
    """Class the helps us extract a bunch of information about a function."""

    # TODO: we should have a bunch of unit tests for this
    # TODO: if the function is declared in a local scope, this function still "works": we should throw an exception
    def __init__(self, f, serialized=False, name_override: Optional[str] = None):
        self.raw_f = f
        self.function_name = name_override if name_override is not None else f.__qualname__
        self.signature = inspect.signature(f)
        module = inspect.getmodule(f)

        if getattr(module, "__package__", None) and not serialized:
            # This is a "real" module, eg. examples.logs.f
            # Get the package path
            # Note: __import__ always returns the top-level package.
            module_file = os.path.abspath(module.__file__)
            package_paths = [os.path.abspath(p) for p in __import__(module.__package__).__path__]
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
            self.remote_dir = ROOT_DIR / PurePosixPath(module.__package__.split(".")[0])
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
            self.type = FunctionInfoType.PACKAGE
            self.serialized_function = None
        elif hasattr(module, "__file__") and not serialized:
            # This generally covers the case where it's invoked with
            # python foo/bar/baz.py
            self.file = inspect.getfile(f)
            self.module_name = inspect.getmodulename(self.file)
            self.base_dir = os.path.dirname(self.file)
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
            self.type = FunctionInfoType.FILE
            self.serialized_function = None
        else:
            self.module_name = None
            self.base_dir = os.path.abspath("")  # get current dir
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_SERIALIZED
            self.serialized_function = serialize(self.raw_f)
            logger.debug(f"Serializing {self.raw_f.__qualname__}, size is {len(self.serialized_function)}")
            if serialized:
                self.type = FunctionInfoType.SERIALIZED
            else:
                self.type = FunctionInfoType.NOTEBOOK

        if self.definition_type == api_pb2.Function.DEFINITION_TYPE_FILE:
            # Sanity check that this function is defined in global scope
            # Unfortunately, there's no "clean" way to do this in Python
            if not is_global_function(f.__qualname__):
                raise LocalFunctionError(
                    "Modal can only import functions defined in global scope unless they are `serialized=True`"
                )

    def get_mounts(self) -> Dict[str, _Mount]:
        if self.type == FunctionInfoType.PACKAGE:
            mounts = {
                self.base_dir: _Mount(
                    local_dir=self.base_dir,
                    remote_dir=self.remote_dir,
                    recursive=True,
                    condition=package_mount_condition,
                )
            }
        elif self.type == FunctionInfoType.FILE:
            mounts = {
                self.file: _Mount(
                    local_file=self.file,
                    remote_dir=ROOT_DIR,
                )
            }
        elif self.type == FunctionInfoType.NOTEBOOK:
            # Don't auto-mount anything for notebooks.
            return {}
        else:
            mounts = {}

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
            if getattr(m, "__package__", None) and getattr(m, "__path__", None):
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
                    remote_dir = ROOT_DIR / PurePosixPath(*m.__name__.split("."))
                    mounts[path] = _Mount(
                        local_dir=path,
                        remote_dir=remote_dir,
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
                dirpath = PurePosixPath(Path(os.path.dirname(path)).resolve().as_posix())
                try:
                    relpath = dirpath.relative_to(Path(self.base_dir).resolve().as_posix())
                except ValueError:
                    # TODO(elias) some kind of heuristic for how to handle things outside of the cwd?
                    continue

                if relpath != PurePosixPath("."):
                    remote_dir = ROOT_DIR / relpath
                else:
                    remote_dir = ROOT_DIR
                mounts[path] = _Mount(
                    local_file=path,
                    remote_dir=remote_dir,
                )
        return filter_safe_mounts(mounts)

    def get_tag(self):
        return self.function_name

    def is_nullary(self):
        return all(param.default is not param.empty for param in self.signature.parameters.values())


def load_function_from_module(module, qual_name):
    # The function might be defined inside a class scope (e.g mymodule.MyClass.f)
    objs: list[Any] = [module]
    if not is_global_function(qual_name):
        raise LocalFunctionError("Attempted to load a function defined in a function scope")

    for path in qual_name.split("."):
        # if a serialized function is defined within a function scope
        # we can't load it from the module and detect a class
        objs.append(getattr(objs[-1], path))

    # If this function is defined on a class, return that too
    cls: Optional[Type] = None
    fun: Callable = objs[-1]
    if len(objs) >= 3:
        cls = objs[-2]

    return cls, fun
