import inspect
import os
import sys
from typing import List

import cloudpickle

from modal_proto import api_pb2

from .config import logger
from .mount import _Mount

ROOT_DIR = "/root"


def package_mount_condition(filename):
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
            package_path = __import__(module.__package__).__path__
            # TODO: we should handle the array case, https://stackoverflow.com/questions/2699287/what-is-path-useful-for
            assert len(package_path) == 1
            (self.package_path,) = package_path
            self.module_name = module.__spec__.name
            self.remote_dir = os.path.join(ROOT_DIR, module.__package__.split(".")[0])
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_PACKAGE
        elif hasattr(module, "__file__") and not serialized:
            # This generally covers the case where it's invoked with
            # python foo/bar/baz.py
            self.file = os.path.abspath(module.__file__)
            self.module_name = os.path.splitext(os.path.basename(self.file))[0]
            self.package_path = os.path.dirname(self.file)
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_SCRIPT
        else:
            # Use cloudpickle. Used when working w/ Jupyter notebooks.
            self.function_serialized = cloudpickle.dumps(f)
            logger.debug(f"Serializing {f.__qualname__}, size is {len(self.function_serialized)}")
            self.module_name = None
            self.package_path = os.path.abspath("")  # get current dir
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_SERIALIZED

    def create_mounts(self, app) -> List[_Mount]:
        if self.definition_type == api_pb2.Function.DEFINITION_TYPE_PACKAGE:
            return [
                _Mount(
                    app=app,
                    local_dir=self.package_path,
                    remote_dir=self.remote_dir,
                    recursive=True,
                    condition=package_mount_condition,
                )
            ]
        else:
            mounts = [
                _Mount(
                    app=app,
                    local_file=self.file,
                    remote_dir=ROOT_DIR,
                )
            ]

            for m in sys.modules.values():
                if getattr(m, "__path__", None):
                    for path in m.__path__:
                        if path.startswith(self.package_path) and not path.startswith(sys.prefix):
                            relpath = os.path.relpath(path, self.package_path)
                            mounts.append(
                                _Mount(
                                    app=app,
                                    local_dir=path,
                                    remote_dir=os.path.join(ROOT_DIR, relpath),
                                    condition=package_mount_condition,
                                    recursive=True,
                                )
                            )
                elif hasattr(m, "__file__"):
                    path = m.__file__
                    if path != self.file and path.startswith(self.package_path) and not path.startswith(sys.prefix):
                        relpath = os.path.relpath(os.path.dirname(path), self.package_path)
                        mounts.append(
                            _Mount(
                                app=app,
                                local_file=path,
                                remote_dir=os.path.join(ROOT_DIR, relpath),
                            )
                        )
        return mounts

    def get_tag(self):
        return f"{self.module_name}.{self.function_name}"

    def is_nullary(self):
        return all(param.default is not param.empty for param in self.signature.parameters.values())
