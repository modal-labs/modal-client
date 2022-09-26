import importlib
import inspect
import os
import sys
from importlib import import_module
from pathlib import Path

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
    """Returns a list of tuples [(module, path, condition)] describing how to mount a given module."""

    file_formats = get_file_formats(module)
    if set(BINARY_FORMATS) & set(file_formats):
        raise Exception(f"{module} can't be mounted because it contains a binary file.")

    m = import_module(module)

    if getattr(m, "__path__", None):
        return [(module, path, module_mount_condition) for path in m.__path__]
    else:
        # Individual file
        filename = m.__file__
        return [(module, os.path.dirname(filename), lambda f: os.path.basename(f) == os.path.basename(filename))]


def import_stub_by_ref(stub_ref: str):
    if stub_ref.find("::") > 1:
        import_path, var_name = stub_ref.split("::")
    elif stub_ref.find(":") > 1:  # don't catch windows abs paths, e.g. C:\foo\bar
        import_path, var_name = stub_ref.split(":")
    else:
        import_path, var_name = stub_ref, "stub"

    if ".py" in import_path:
        # walk to the closest python package in the path and add that to the path
        # before importing, in case of imports etc. of other modules in that package
        # are needed
        file_path = os.path.abspath(import_path)

        # Let's first assume this is not part of any package
        module_name = inspect.getmodulename(import_path)

        # Look for any __init__.py in a parent directory and maybe change the module name
        directory = Path(file_path).parent
        module_path = [inspect.getmodulename(file_path)]
        while directory.parent != directory:
            parent = directory.parent
            module_path.append(directory.name)
            if (directory / "__init__.py").exists():
                # We identified a package, let's store a new module name
                module_name = ".".join(reversed(module_path))
            directory = parent

        # Import the module - see https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        sys.path.append(os.getcwd())
        module = importlib.import_module(import_path)

    stub = getattr(module, var_name)
    return stub
