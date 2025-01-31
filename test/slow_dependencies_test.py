# Copyright Modal Labs 2025
import subprocess
import sys


def test_slow_dependencies_local(supports_dir):
    # Make sure that "import modal" doesn't load some big dependencies like aiohttp
    subprocess.check_output([sys.executable, supports_dir / "slow_dependencies_local.py"])


def test_slow_dependencies_container(supports_dir):
    # Make sure that "import modal._container_entrypoint" doesn't load some big dependencies like aiohttp
    subprocess.check_output([sys.executable, supports_dir / "slow_dependencies_container.py"])
