# Copyright Modal Labs 2022
import os
import pytest
from importlib import metadata

from packaging.version import Version

import modal


def test_version():
    mod_version = modal.__version__
    assert Version(mod_version) > Version("0.0.0")

    if os.environ.get("TEST_SRCDIR"):
        pytest.skip("importlib.metadata not available for source packages in Bazel sandbox")

    pkg_version = metadata.version("modal")
    assert Version(pkg_version) > Version("0.0.0")
    assert mod_version == pkg_version
