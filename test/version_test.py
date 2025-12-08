# Copyright Modal Labs 2022
from importlib.metadata import version as get_pkg_version

from packaging.version import parse as parse_version

import modal


def test_version():
    mod_version = modal.__version__
    pkg_version = get_pkg_version("modal")

    assert parse_version(mod_version) > parse_version("0.0.0")
    assert parse_version(pkg_version) > parse_version("0.0.0")

    assert mod_version == pkg_version
