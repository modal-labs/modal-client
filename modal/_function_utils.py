# Copyright Modal Labs 2022
import inspect
import os
import site
import sys
import sysconfig
import typing
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Type

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
    def __init__(self, f, serialized=False, name_override: Optional[str] = None, cls: Optional[Type] = None):
        self.raw_f = f
        self.cls = cls

        if name_override is not None:
            self.function_name = name_override
        elif f.__qualname__ != f.__name__ and not serialized:
            # Class function.
            if len(f.__qualname__.split(".")) > 2:
                raise InvalidError("@stub.cls classes must be defined in global scope")
            self.function_name = f"{cls.__name__}.{f.__name__}"
        else:
            self.function_name = f.__qualname__

        self.signature = inspect.signature(f)

        # If it's a cls, the @method could be defined in a base class in a different file.
        if cls is not None:
            module = inspect.getmodule(cls)
        else:
            module = inspect.getmodule(f)

        if getattr(module, "__package__", None) and not serialized:
            # This is a "real" module, eg. examples.logs.f
            # Get the package path
            # Note: __import__ always returns the top-level package.
            self.file = os.path.abspath(module.__file__)
            package_paths = set([os.path.abspath(p) for p in __import__(module.__package__).__path__])
            # There might be multiple package paths in some weird cases
            base_dirs = [
                base_dir for base_dir in package_paths if os.path.commonpath((base_dir, self.file)) == base_dir
            ]

            if not base_dirs:
                logger.info(f"Module files: {self.file}")
                logger.info(f"Package paths: {package_paths}")
                logger.info(f"Base dirs: {base_dirs}")
                raise Exception("Wasn't able to find the package directory!")
            elif len(base_dirs) > 1:
                # Base_dirs should all be prefixes of each other since they all contain `module_file`.
                base_dirs.sort(key=len)

            self.base_dir = base_dirs[0]
            self.module_name = module.__spec__.name
            self.remote_dir = ROOT_DIR / PurePosixPath(module.__package__.split(".")[0])
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
            self.type = FunctionInfoType.PACKAGE
        elif hasattr(module, "__file__") and not serialized:
            # This generally covers the case where it's invoked with
            # python foo/bar/baz.py

            # If it's a cls, the @method could be defined in a base class in a different file.
            if cls is not None:
                self.file = os.path.abspath(inspect.getfile(cls))
            else:
                self.file = os.path.abspath(inspect.getfile(f))
            self.module_name = inspect.getmodulename(self.file)
            self.base_dir = os.path.dirname(self.file)
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
            self.type = FunctionInfoType.FILE
        else:
            self.module_name = None
            self.base_dir = os.path.abspath("")  # get current dir
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_SERIALIZED
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

    def is_serialized(self) -> bool:
        return self.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED

    def serialized_function(self) -> bytes:
        # Note: this should only be called from .load() and not at function decoration time
        #       otherwise the serialized function won't have access to variables/side effect
        #        defined after it in the same file
        assert self.is_serialized()
        serialized_bytes = serialize(self.raw_f)
        logger.debug(f"Serializing {self.raw_f.__qualname__}, size is {len(serialized_bytes)}")
        return serialized_bytes

    def get_globals(self):
        from cloudpickle.cloudpickle import _extract_code_globals

        func = self.raw_f
        f_globals_ref = _extract_code_globals(func.__code__)
        f_globals = {k: func.__globals__[k] for k in f_globals_ref if k in func.__globals__}
        return f_globals

    def get_mounts(self) -> Dict[str, _Mount]:
        """
        Includes:
        * Implicit mount of the function itself (the module or package that the function is part of)
        * "Auto mounted" mounts, i.e. all mounts in sys.modules that are *not* installed in site-packages.
            These are typically local modules which are imported but not part of the running package

        Does not include:
        * Client mount
        * Explicit mounts added to the stub or function declaration
        """
        if self.type == FunctionInfoType.NOTEBOOK:
            # Don't auto-mount anything for notebooks.
            return {}

        if config.get("automount"):
            mounts = self._get_auto_mounts()
        else:
            mounts = {}

        # make sure the function's own entrypoint is included:
        if self.type == FunctionInfoType.PACKAGE and config.get("automount"):
            mounts[self.base_dir] = _Mount.from_local_dir(
                self.base_dir,
                remote_path=self.remote_dir,
                recursive=True,
                condition=package_mount_condition,
            )
        elif self.definition_type == api_pb2.Function.DEFINITION_TYPE_FILE:
            remote_path = ROOT_DIR / Path(self.file).name
            mounts[self.file] = _Mount.from_local_file(
                self.file,
                remote_path=remote_path,
            )

        return filter_safe_mounts(mounts)

    def _get_auto_mounts(self):
        # Auto-mount local modules that have been imported in global scope.
        # This may or may not include the "entrypoint" of the function as well, depending on how modal is invoked
        # Note: sys.modules may change during the iteration
        mounts = {}
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
                    mounts[path] = _Mount.from_local_dir(
                        path,
                        remote_path=remote_dir,
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
                    remote_path = ROOT_DIR / relpath / Path(path).name
                else:
                    remote_path = ROOT_DIR / Path(path).name

                mounts[path] = _Mount.from_local_file(
                    path,
                    remote_path=remote_path,
                )
        return mounts

    def get_tag(self):
        return self.function_name

    def is_nullary(self):
        return all(param.default is not param.empty for param in self.signature.parameters.values())
