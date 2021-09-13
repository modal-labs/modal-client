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


def get_dependencies_recursive(module_name, visited=set()):
    visited.add(module_name)
    for req_name in requires(module_name) or []:
        try:
            req = Requirement(req_name)
            # skip reqs with markers e.g. python_version < 3.2
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            if req.name not in visited:
                get_dependencies_recursive(req.name, visited)
        except Exception as e:
            print(f"Failed getting deps for {req}: {repr(e)}")
    return list(visited)


def package_mount_condition(f):
    return not any([f.endswith(".pyc"), f.startswith(".")])


BINARY_FORMATS = ["so", "S", "s", "asm"]  # TODO


def get_mount_info(package_name, module_name):
    """Returns a tuple (module_name, path, condition) in order to mount a given module."""

    file_formats = get_file_formats(package_name)
    logger.info(f"{package_name}: {file_formats}")
    for bf in BINARY_FORMATS:
        if bf in file_formats:
            logger.info(f"Skipping {package_name} because it contains a binary .{bf} file.")
            return []

    try:
        m = import_module(module_name)
    except Exception as e:
        logger.exception(repr(e))
        return []

    if getattr(m, "__path__", None):
        return [(module_name, path, package_mount_condition) for path in m.__path__]
    else:
        # Individual file
        filename = m.__file__
        return [(module_name, os.path.dirname(filename), lambda f: os.path.basename(f) == os.path.basename(filename))]


def get_package_deps_mount_info(package_name):
    """Get mount info for all recursive dependencies of the given package name (including self)."""

    all_deps = get_dependencies_recursive(package_name)

    result = []

    for d in all_deps:
        module_name = requirement_to_module_name(d)
        result.extend(get_mount_info(d, module_name))

    return result
