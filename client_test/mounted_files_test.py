import os
import pytest
import subprocess
import sys
from pathlib import Path


@pytest.fixture(scope="module")
def test_dir(request):
    """Absolute path to directory containing test file."""
    root_dir = Path(request.config.rootdir)
    test_dir = Path(os.getenv("PYTEST_CURRENT_TEST")).parent
    return root_dir / test_dir


def test_function_info_script(test_dir):
    p = subprocess.run([sys.executable, "supports/mounted_files.py"], capture_output=True, cwd=test_dir)
    files = p.stdout.decode("utf-8").splitlines()

    assert len(files) == 1
    assert "client/client_test/supports/mounted_files.py" in files[0]


def test_function_info_package(test_dir):
    p = subprocess.run([sys.executable, "-m", "supports.mounted_files"], cwd=test_dir, capture_output=True)
    files = p.stdout.decode("utf-8").splitlines()

    assert len(files) == 1
    assert "client/client_test/supports/mounted_files.py" in files[0]
