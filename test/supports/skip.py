# Copyright Modal Labs 2022

import platform
import pytest
import sys


def skip_windows(msg: str):
    return pytest.mark.skipif(
        platform.system() == "Windows",
        reason=msg,
    )


skip_windows_unix_socket = skip_windows("Windows doesn't have UNIX sockets")


def skip_old_py(msg: str, min_version: tuple):
    return pytest.mark.skipif(sys.version_info < min_version, reason=msg)
