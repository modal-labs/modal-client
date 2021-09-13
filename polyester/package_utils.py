import os

from importlib import import_module
from importlib.metadata import requires, files
from packaging.requirements import Requirement

from .config import logger


def get_file_formats(package):
    endings = [str(p).split(".")[-1] for p in files(package) if "." in str(p)]
    return list(set(endings))


# HACK to find mapping b/w requirement name and actual module
# e.g. 'grpcio' to 'grpc'
def requirement_to_module_name(package):
    for path_obj in files(package):
        filename = str(path_obj)
        if filename.endswith((".py")):
            parts = os.path.splitext(filename)[0].split(os.sep)
            if filename.endswith(".py") and parts[-1] == "__init__":
                return parts[-2]
    return package


def package_mount_condition(f):
    return not any([f.endswith(".pyc"), f.startswith(".")])


BINARY_FORMATS = ["so", "S", "s", "asm"]  # TODO


def get_mount_info(package_name, module_name):
    """Returns a list of tuples [(module_name, path, condition)] describing how to mount a given module."""

    file_formats = get_file_formats(package_name)
    logger.info(f"{package_name}: {file_formats}")
    if set(BINARY_FORMATS) & set(file_formats):
        raise Exception(f"{package_name} can't be mounted because it contains a binary file.")

    m = import_module(module_name)

    if getattr(m, "__path__", None):
        return [(module_name, path, package_mount_condition) for path in m.__path__]
    else:
        # Individual file
        filename = m.__file__
        return [(module_name, os.path.dirname(filename), lambda f: os.path.basename(f) == os.path.basename(filename))]


def get_package_deps_mount_info(package_name):
    """Get mount info for all recursive dependencies of the given package name (including self)."""
    module_name = requirement_to_module_name(package_name)
    return get_mount_info(package_name, module_name)
