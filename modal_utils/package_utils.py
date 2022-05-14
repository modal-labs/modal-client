import hashlib
import importlib
import os
import sys
from importlib import import_module

from importlib_metadata import PackageNotFoundError, files

from .logger import logger


def _get_sha256_hex_from_content(content):
    return hashlib.sha256(content).hexdigest()


def get_sha256_hex_from_filename(filename, rel_filename):
    # Somewhat CPU intensive, so we run it in a thread/process
    content = open(filename, "rb").read()
    return filename, rel_filename, content, _get_sha256_hex_from_content(content)


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
    logger.info(f"{module}: {file_formats}")
    if set(BINARY_FORMATS) & set(file_formats):
        raise Exception(f"{module} can't be mounted because it contains a binary file.")

    m = import_module(module)

    if getattr(m, "__path__", None):
        return [(module, path, module_mount_condition) for path in m.__path__]
    else:
        # Individual file
        filename = m.__file__
        return [(module, os.path.dirname(filename), lambda f: os.path.basename(f) == os.path.basename(filename))]


def import_app_by_ref(app_ref: str):
    root_dir = os.getcwd()
    if ".py" in app_ref:
        # walk to the closest python package in the path and add that to the path
        # before importing, in case of imports etc. of other modules in that package
        # are needed
        file_path, var_part = app_ref.split(".py")
        module_segments = file_path.split("/")
        for path_segment in module_segments.copy()[:-1]:
            if os.path.exists("__init__.py"):  # is package
                break
            root_dir += f"/{path_segment}"
            module_segments = module_segments[1:]

        import_path = ".".join(module_segments)
        var_name = var_part.lstrip(":")
    else:
        if "::" in app_ref:
            import_path, var_name = app_ref.split("::")
        elif ":" in app_ref:
            import_path, var_name = app_ref.split(":")
        else:
            import_path, var_name = app_ref, "app"

    sys.path.append(root_dir)
    var_name = var_name or "app"
    module = importlib.import_module(import_path)
    app = getattr(module, var_name)
    return app
