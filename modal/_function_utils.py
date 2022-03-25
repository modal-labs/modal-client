import inspect
import os

import cloudpickle

from modal_proto import api_pb2

from .config import logger


class FunctionInfo:
    """Class the helps us extracting a bunch of information about a function."""

    # TODO: we should have a bunch of unit tests for this
    # TODO: if the function is declared in a local scope, this function still "works": we should throw an exception
    def __init__(self, f):
        self.function_name = f.__qualname__
        self.function_serialized = None
        self.signature = inspect.signature(f)
        module = inspect.getmodule(f)
        if module.__package__:
            # This is a "real" module, eg. examples.logs.f
            # Get the package path
            # Note: __import__ always returns the top-level package.
            package_path = __import__(module.__package__).__path__
            # TODO: we should handle the array case, https://stackoverflow.com/questions/2699287/what-is-path-useful-for
            assert len(package_path) == 1
            (self.package_path,) = package_path
            self.module_name = module.__spec__.name
            self.recursive = True
            self.remote_dir = "/root/" + module.__package__.split(".")[0]  # TODO: don't hardcode /root
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
        elif hasattr(module, "__file__"):
            # This generally covers the case where it's invoked with
            # python foo/bar/baz.py
            module_fn = os.path.abspath(module.__file__)
            self.module_name = os.path.splitext(os.path.basename(module_fn))[0]
            self.package_path = os.path.dirname(module_fn)
            self.recursive = False  # Just pick out files in the same directory
            self.remote_dir = "/root"  # TODO: don't hardcore /root
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_FILE
        else:
            # Use cloudpickle. Used when working w/ Jupyter notebooks.
            self.function_serialized = cloudpickle.dumps(f)
            logger.info(f"Serializing {f.__qualname__}, size is {len(self.function_serialized)}")
            self.module_name = None
            self.package_path = os.path.abspath("")  # get current dir
            self.recursive = False  # Just pick out files in the same directory
            self.remote_dir = "/root"  # TODO: don't hardcore /root
            self.definition_type = api_pb2.Function.DEFINITION_TYPE_SERIALIZED

        self.condition = lambda filename: os.path.splitext(filename)[1] in [".py", ".ipynb"]

    def get_tag(self):
        return f"{self.module_name}.{self.function_name}"

    def is_nullary(self):
        return all(param.default is not param.empty for param in self.signature.parameters.values())
