# Copyright Modal Labs 2022
import dataclasses
import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Optional

from importlib_metadata import PackageNotFoundError, files

import modal.exception


def get_file_formats(module):
    try:
        endings = [str(p).split(".")[-1] for p in files(module) if "." in str(p)]
        return list(set(endings))
    except PackageNotFoundError:
        return []


def module_mount_condition(f):
    return not any([f.endswith(".pyc"), os.path.basename(f).startswith(".")])


BINARY_FORMATS = ["so", "S", "s", "asm"]  # TODO


def get_module_mount_info(module: str):
    """Returns a list of tuples [(is_package, path, condition)] describing how to mount a given module."""

    file_formats = get_file_formats(module)
    if set(BINARY_FORMATS) & set(file_formats):
        raise Exception(f"{module} can't be mounted because it contains a binary file.")

    spec = importlib.util.find_spec(module)

    if spec is None:
        return []
    elif spec.submodule_search_locations:
        return [(True, path, module_mount_condition) for path in spec.submodule_search_locations]
    else:
        # Individual file
        filename = spec.origin
        return [(False, filename, lambda f: os.path.basename(f) == os.path.basename(filename))]


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

    obj_path = import_ref.object_path or DEFAULT_STUB_NAME  # get variable named "stub" by default

    return get_by_object_path(module, obj_path)


def get_by_object_path(obj: Any, obj_path: str):
    # attempt to resolve '.'-delimited object path in a parent object
    # If one path segment doesn't exist, try to instead reference *items*
    # with dot-delimited keys until a matching object is found
    #
    # Note: this is eager, so no backtracking is performed in case an
    # earlier match fails at some later point in the path expansion
    orig_obj = obj
    prefix = ""
    for segment in obj_path.split("."):
        attr = prefix + segment
        try:
            if "." in attr:
                obj = obj[attr]
            else:
                obj = getattr(obj, attr)
        except (AttributeError, KeyError):
            prefix = f"{prefix}{segment}."
        else:
            prefix = ""

    if prefix:
        raise NoSuchObject(f"No object {obj_path} could be found in module {orig_obj}")

    return obj
