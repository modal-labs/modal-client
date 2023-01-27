"""Utilities for CLI references to Modal entities

For example, the function reference of `modal run some_file.py::stub.foo_func`
or the stub lookup of `modal deploy some_file.py`
"""

import dataclasses
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Optional

import modal
from modal.stub import _Stub


@dataclasses.dataclass
class ImportRef:
    file_or_module: str
    object_path: Optional[str]


def parse_import_ref(object_ref: str) -> ImportRef:
    if object_ref.find("::") > 1:
        file_or_module, object_path = object_ref.split("::", 1)
    else:
        file_or_module, object_path = object_ref, None

    return ImportRef(file_or_module, object_path)


class NoSuchObject(modal.exception.NotFoundError):
    pass


DEFAULT_STUB_NAME = "stub"


def import_object(import_ref: ImportRef):
    if "" not in sys.path:
        # This seems to happen when running from a CLI
        sys.path.insert(0, "")
    import_path = import_ref.file_or_module
    if ".py" in import_path:
        # walk to the closest python package in the path and add that to the path
        # before importing, in case of imports etc. of other modules in that package
        # are needed

        # Let's first assume this is not part of any package
        module_name = inspect.getmodulename(import_path)

        # Look for any __init__.py in a parent directory and maybe change the module name
        directory = Path(import_path).parent
        module_path = [inspect.getmodulename(import_path)]
        while directory.parent != directory:
            parent = directory.parent
            module_path.append(directory.name)
            if (directory / "__init__.py").exists():
                # We identified a package, let's store a new module name
                module_name = ".".join(reversed(module_path))
            directory = parent

        # Import the module - see https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location(module_name, import_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(import_path)

    return module


def get_by_object_path(obj: Any, obj_path: str):
    # Try to evaluate a `.`-delimited object path in a Modal context
    # With the caveat that some object names can actually have `.` in their name (lifecycled methods' tags)

    # Note: this is eager, so no backtracking is performed in case an
    # earlier match fails at some later point in the path expansion
    orig_obj = obj
    prefix = ""
    for segment in obj_path.split("."):
        attr = prefix + segment
        try:
            if isinstance(obj, _Stub):
                if attr in obj.registered_entrypoints:
                    # local entrypoints are not on stub blueprint
                    obj = obj.registered_functions[attr]
                    continue
            obj = getattr(obj, attr)

        except Exception:
            prefix = f"{prefix}{segment}."
        else:
            prefix = ""

    if prefix:
        raise NoSuchObject(f"No object {obj_path} could be found in module {orig_obj}")

    return obj
