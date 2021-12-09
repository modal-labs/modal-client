import inspect
import json
import os

import cloudpickle

from .config import logger
from .mount import Mount
from .proto import api_pb2


class FunctionInfo:
    """Class the helps us extracting a bunch of information about a function."""

    # TODO: we should have a bunch of unit tests for this
    # TODO: if the function is declared in a local scope, this function still "works": we should throw an exception
    def __init__(self, f):
        self.function_name = f.__name__
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
            self.recursive_upload = True
            self.remote_dir = "/root/" + module.__package__.split(".")[0]  # TODO: don't hardcode /root
            self.definition_type = api_pb2.Function.DefinitionType.FILE
        elif hasattr(module, "__file__"):
            # This generally covers the case where it's invoked with
            # python foo/bar/baz.py
            self.module_name = os.path.splitext(os.path.basename(module.__file__))[0]
            self.package_path = os.path.dirname(module.__file__)
            self.recursive_upload = False  # Just pick out files in the same directory
            self.remote_dir = "/root"  # TODO: don't hardcore /root
            self.definition_type = api_pb2.Function.DefinitionType.FILE
        else:
            # Use cloudpickle. Used when working w/ Jupyter notebooks.
            self.function_serialized = cloudpickle.dumps(f)
            logger.info(f"Serializing {f.__name__}, size is {len(self.function_serialized)}")
            self.module_name = None
            self.package_path = os.path.abspath("")  # get current dir
            self.recursive_upload = False  # Just pick out files in the same directory
            self.remote_dir = "/root"  # TODO: don't hardcore /root
            self.definition_type = api_pb2.Function.DefinitionType.SERIALIZED

    def get_mount(self):
        return Mount(
            local_dir=self.package_path,
            remote_dir=self.remote_dir,
            condition=lambda filename: os.path.splitext(filename)[1] in [".py", ".ipynb"],
            recursive=self.recursive_upload,
        )

    def get_tag(self, args, kwargs):
        # TODO: merge code with FunctionInfo, get module name too
        # TODO: break this out into a utility function
        if args is not None:
            args = self.signature.bind(*args, **kwargs)
            args.apply_defaults()
            args = list(args.arguments.values())
            args = json.dumps(args)
            args = args[1:-1]  # Shave off the outer []
            return f"{self.module_name}.{self.function_name}({args})"
        else:
            return f"{self.module_name}.{self.function_name}"
            tag = fun.__name__
