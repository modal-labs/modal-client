# Copyright Modal Labs 2022

import platform

import pytest

# TODO(erikbern): there's a few other reasons we skip windows (see eg shared_volume_test.py)
skip_windows = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Windows doesn't have UNIX sockets",
)
