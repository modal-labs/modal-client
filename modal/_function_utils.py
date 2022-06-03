import inspect
import os
import sys
from typing import Dict

import cloudpickle

from modal_proto import api_pb2

from .config import logger
from .mount import _Mount

ROOT_DIR = "/root"


def package_mount_condition(filename):
    if filename.startswith(sys.prefix):
        return False

    return os.path.splitext(filename)[1] in [".py"]


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
                raise Exception(
                    f"Wasn't able to identify the base directory! {module_file=} {package_paths=} {base_dirs=}"
                )
            (self.base_dir,) = base_dirs
            self.module_name = module.__spec__.name
            self.remote_dir = os.path.join(ROOT_DIR, module.__package__.split(".")[0])
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
            self.is_package = True
        elif hasattr(module, "__file__") and not serialized:
            # This generally covers the case where it's invoked with
            # python foo/bar/baz.py
            self.file = os.path.abspath(module.__file__)
            self.module_name = os.path.splitext(os.path.basename(self.file))[0]
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
            return {
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

            packages = set()

            # Note: sys.modules may change during the iteration
            modules = list(sys.modules.values())
            for m in modules:
                if getattr(m, "__package__", None):
                    for path in __import__(m.__package__).__path__:
                        if path in packages or not path.startswith(self.base_dir) or path.startswith(sys.prefix):
                            continue

                        packages.add(path)
                        relpath = os.path.relpath(path, self.base_dir)
                        mounts[path] = _Mount(
                            local_dir=path,
                            remote_dir=os.path.join(ROOT_DIR, relpath),
                            condition=package_mount_condition,
                            recursive=True,
                        )
                elif getattr(m, "__file__", None):
                    path = m.__file__
                    if path == self.file or not path.startswith(self.base_dir) or path.startswith(sys.prefix):
                        continue
                    relpath = os.path.relpath(os.path.dirname(path), self.base_dir)
                    mounts[path] = _Mount(
                        local_file=path,
                        remote_dir=os.path.join(ROOT_DIR, relpath),
                    )
            return mounts
        else:
            return {}

    def get_tag(self):
        return f"{self.module_name}.{self.function_name}"

    def is_nullary(self):
        return all(param.default is not param.empty for param in self.signature.parameters.values())
