import hashlib
import os
from importlib import import_module

from importlib_metadata import PackageNotFoundError, files

from .logger import logger


def _get_sha256_hex_from_content(content):
    m = hashlib.sha256()
    m.update(content)
    return m.hexdigest()


def get_sha256_hex_from_filename(filename, rel_filename):
    # Somewhat CPU intensive, so we run it in a thread/process
    content = open(filename, "rb").read()
    return filename, rel_filename, _get_sha256_hex_from_content(content)


def get_file_formats(module):
    try:
        endings = [str(p).split(".")[-1] for p in files(module) if "." in str(p)]
        return list(set(endings))
    except PackageNotFoundError:
        return []


def module_mount_condition(f):
    return not any([f.endswith(".pyc"), f.startswith(".")])


BINARY_FORMATS = ["so", "S", "s", "asm"]  # TODO


def get_module_mount_info(module):
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
