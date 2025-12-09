# Copyright Modal Labs 2022
from importlib import metadata

from packaging.version import Version

import modal


def test_version():
    mod_version = modal.__version__
    pkg_version = metadata.version("modal")

    assert Version(mod_version) > Version("0.0.0")
    assert Version(pkg_version) > Version("0.0.0")

    assert mod_version == pkg_version
