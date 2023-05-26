# Copyright Modal Labs 2022
import pkg_resources

import modal


def test_version():
    mod_version = modal.__version__
    pkg_version = pkg_resources.require("modal-client")[0].version

    assert pkg_resources.parse_version(mod_version) > pkg_resources.parse_version("0.0.0")
    assert pkg_resources.parse_version(pkg_version) > pkg_resources.parse_version("0.0.0")

    assert mod_version == pkg_version
