# Copyright Modal Labs 2022
import importlib
import importlib.util
import os

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
