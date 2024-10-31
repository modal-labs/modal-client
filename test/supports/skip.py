# Copyright Modal Labs 2022
import os
import platform
import pytest
import sys


def skip_windows(msg: str):
    return pytest.mark.skipif(platform.system() == "Windows", reason=msg)


def skip_macos(msg: str):
    return pytest.mark.skipif(platform.system() == "Darwin", reason=msg)


skip_windows_unix_socket = skip_windows("Windows doesn't have UNIX sockets")


def skip_old_py(msg: str, min_version: tuple):
    return pytest.mark.skipif(sys.version_info < min_version, reason=msg)


skip_github_non_linux = pytest.mark.skipif(
    (os.environ.get("GITHUB_ACTIONS") == "true" and platform.system() != "Linux"),
    reason="containers only have to run on linux.",
)
