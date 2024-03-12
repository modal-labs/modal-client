# Copyright Modal Labs 2022
import inspect
import os
import site
import sys
import sysconfig
import typing
from collections import deque
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Callable, List, Optional, Set, Type

from modal_proto import api_pb2

from .._serialization import serialize
from ..config import config, logger
from ..exception import InvalidError, ModuleNotMountable
from ..mount import ROOT_DIR, _Mount
from ..object import Object

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


def entrypoint_only_package_mount_condition(entrypoint_file):
    entrypoint_path = Path(entrypoint_file)

    def inner(filename):
        path = Path(filename)
        if path == entrypoint_path:
            return True
        if path.name == "__init__.py" and path.parent in entrypoint_path.parents:
            # ancestor __init__.py are included
            return True
        return False

    return inner


def _is_modal_path(remote_path: PurePosixPath):
    path_prefix = remote_path.parts[:3]
    remote_python_paths = [("/", "root"), ("/", "pkg")]
    for base in remote_python_paths:
        is_modal_path = path_prefix in [
            base + ("modal",),
            base + ("modal_proto",),
            base + ("modal_version",),
            base + ("synchronicity",),
        ]
        if is_modal_path:
            return True
    return False


def is_global_function(function_qual_name):
    return "<locals>" not in function_qual_name.split(".")


def is_async(function):
    # TODO: this is somewhat hacky. We need to know whether the function is async or not in order to
    # coerce the input arguments to the right type. The proper way to do is to call the function and
    # see if you get a coroutine (or async generator) back. However at this point, it's too late to
    # coerce the type. For now let's make a determination based on inspecting the function definition.
    # This sometimes isn't correct, since a "vanilla" Python function can return a coroutine if it
    # wraps async code or similar. Let's revisit this shortly.
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        return True
    elif inspect.isfunction(function) or inspect.isgeneratorfunction(function):
        return False
    else:
        raise RuntimeError(f"Function {function} is a strange type {type(function)}")


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
            self.file = os.path.abspath(inspect.getfile(module))
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
        from .._vendor.cloudpickle import _extract_code_globals

        func = self.raw_f
        f_globals_ref = _extract_code_globals(func.__code__)
        f_globals = {k: func.__globals__[k] for k in f_globals_ref if k in func.__globals__}
        return f_globals

    def get_entrypoint_mount(self) -> List[_Mount]:
        """
        Includes:
        * Implicit mount of the function itself (the module or package that the function is part of)

        Does not include:
        * Client mount
        * Explicit mounts added to the stub or function declaration
        * "Auto mounted" mounts, i.e. all mounts in sys.modules that are *not* installed in site-packages.
            These are typically local modules which are imported but not part of the running package

        """
        if self.type == FunctionInfoType.NOTEBOOK:
            # Don't auto-mount anything for notebooks.
            return []

        # make sure the function's own entrypoint is included:
        if self.type == FunctionInfoType.PACKAGE:
            if config.get("automount"):
                return [_Mount.from_local_python_packages(self.module_name)]
            elif self.definition_type == api_pb2.Function.DEFINITION_TYPE_FILE:
                # mount only relevant file and __init__.py:s
                return [
                    _Mount.from_local_dir(
                        self.base_dir,
                        remote_path=self.remote_dir,
                        recursive=True,
                        condition=entrypoint_only_package_mount_condition(self.file),
                    )
                ]
        elif self.definition_type == api_pb2.Function.DEFINITION_TYPE_FILE:
            remote_path = ROOT_DIR / Path(self.file).name
            if not _is_modal_path(remote_path):
                return [
                    _Mount.from_local_file(
                        self.file,
                        remote_path=remote_path,
                    )
                ]
        return []

    def get_auto_mounts(self) -> typing.List[_Mount]:
        # Auto-mount local modules that have been imported in global scope.
        # This may or may not include the "entrypoint" of the function as well, depending on how modal is invoked
        # Note: sys.modules may change during the iteration
        auto_mounts = []
        top_level_modules = []
        skip_prefixes = set()
        for name, module in sorted(sys.modules.items(), key=lambda kv: len(kv[0])):
            parent = name.rsplit(".")[0]
            if parent and parent in skip_prefixes:
                skip_prefixes.add(name)
                continue
            skip_prefixes.add(name)
            top_level_modules.append((name, module))

        for module_name, module in top_level_modules:
            if module_name.startswith("__"):
                # skip "built in" modules like __main__ and __mp_main__
                # the running function's main file should be included anyway
                continue

            try:
                # at this point we don't know if the sys.modules module should be mounted or not
                potential_mount = _Mount.from_local_python_packages(module_name)
                mount_paths = potential_mount._top_level_paths()
            except ModuleNotMountable:
                # this typically happens if the module is a built-in, has binary components or doesn't exist
                continue

            for local_path, remote_path in mount_paths:
                if any(str(local_path).startswith(p) for p in SYS_PREFIXES) or _is_modal_path(remote_path):
                    # skip any module that has paths in SYS_PREFIXES, or would overwrite the modal Package in the container
                    break
            else:
                auto_mounts.append(potential_mount)

        return auto_mounts

    def get_tag(self):
        return self.function_name

    def is_nullary(self):
        for param in self.signature.parameters.values():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                # variadic parameters are nullary
                continue
            if param.default is param.empty:
                return False
        return True


def get_referred_objects(f: Callable) -> List[Object]:
    """Takes a function and returns any Modal Objects in global scope that it refers to.

    TODO: this does not yet support Object contained by another object,
    e.g. a list of Objects in global scope.
    """
    from ..cls import Cls
    from ..functions import Function

    ret: List[Object] = []
    obj_queue: deque[Callable] = deque([f])
    objs_seen: Set[int] = set([id(f)])
    while obj_queue:
        obj = obj_queue.popleft()
        if isinstance(obj, (Function, Cls)):
            # These are always attached to stubs, so we shouldn't do anything
            pass
        elif isinstance(obj, Object):
            ret.append(obj)
        elif inspect.isfunction(obj):
            try:
                closure_vars = inspect.getclosurevars(obj)
            except ValueError:
                logger.warning(
                    f"Could not inspect closure vars of {f} - referenced global Modal objects may or may not work in that function"
                )
                continue

            for dep_obj in closure_vars.globals.values():
                if id(dep_obj) not in objs_seen:
                    objs_seen.add(id(dep_obj))
                    obj_queue.append(dep_obj)
    return ret


def method_has_params(f: Callable) -> bool:
    """Return True if a method (bound or unbound) has parameters other than self.

    Used for deprecation of @exit() parameters.
    """
    num_params = len(inspect.signature(f).parameters)
    if hasattr(f, "__self__"):
        return num_params > 0
    else:
        return num_params > 1
