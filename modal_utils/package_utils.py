import importlib
import os
import sys
from importlib import import_module

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
    root_dir = os.getcwd()
    if ".py" in stub_ref:
        # walk to the closest python package in the path and add that to the path
        # before importing, in case of imports etc. of other modules in that package
        # are needed
        file_path, var_part = stub_ref.split(".py")
        module_segments = file_path.split("/")
        for path_segment in module_segments.copy()[:-1]:
            if os.path.exists("__init__.py"):  # is package
                break
            root_dir += f"/{path_segment}"
            module_segments = module_segments[1:]

        import_path = ".".join(module_segments)
        var_name = var_part.lstrip(":")
    else:
        if "::" in stub_ref:
            import_path, var_name = stub_ref.split("::")
        elif ":" in stub_ref:
            import_path, var_name = stub_ref.split(":")
        else:
            import_path, var_name = stub_ref, "stub"

    sys.path.append(root_dir)
    var_name = var_name or "stub"
    module = importlib.import_module(import_path)
    stub = getattr(module, var_name)
    return stub
