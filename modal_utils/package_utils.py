# Copyright Modal Labs 2022
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Tuple

from importlib_metadata import PackageNotFoundError, files


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


def parse_stub_ref(stub_ref: str, default_stub_name: str) -> Tuple[str, str]:
    if stub_ref.find("::") > 1:
        import_path, stub_name = stub_ref.split("::")
    elif stub_ref.find(":") > 1:  # don't catch windows abs paths, e.g. C:\foo\bar
        import_path, stub_name = stub_ref.split(":")
    else:
        import_path, stub_name = stub_ref, default_stub_name
    return import_path, stub_name


def import_stub(import_path: str, stub_name: str):
    if "" not in sys.path:
        # This seems to happen when running from a CLI
        sys.path.insert(0, "")

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

    stub = getattr(module, stub_name)
    return stub
