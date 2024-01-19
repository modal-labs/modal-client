# Copyright Modal Labs 2022
import os
import pytest
import subprocess
import sys
from pathlib import Path

import pytest_asyncio

from modal._function_utils import FunctionInfo

from . import helpers
from .supports.skip import skip_windows


@pytest.fixture
def venv_path(tmp_path):
    venv_path = tmp_path
    subprocess.run([sys.executable, "-m", "venv", venv_path, "--copies", "--system-site-packages"], check=True)
    # Install Modal and a tiny package in the venv.
    subprocess.run([venv_path / "bin" / "python", "-m", "pip", "install", "-e", "."], check=True)
    subprocess.run([venv_path / "bin" / "python", "-m", "pip", "install", "--force-reinstall", "six"], check=True)
    yield venv_path


script_path = "pkg_a/script.py"


@pytest.fixture
def supports_dir(test_dir):
    return test_dir / Path("supports")


def f():
    pass


@pytest_asyncio.fixture
async def env_mount_files():
    # If something is installed using pip -e, it will be bundled up as a part of the environment.
    # Those are env-specific so we ignore those as a part of the test
    fn_info = FunctionInfo(f)

    filenames = []
    for _, mount in fn_info.get_mounts().items():
        async for file_info in mount._get_files(mount.entries):
            filenames.append(file_info.mount_filename)

    return filenames


def test_mounted_files_script(servicer, supports_dir, env_mount_files, server_url_env):
    helpers.deploy_stub_externally(servicer, script_path, cwd=supports_dir)
    files = set(servicer.files_name2sha.keys()) - set(env_mount_files)

    # Assert we include everything from `pkg_a` and `pkg_b` but not `pkg_c`:
    assert files == {
        "/root/a.py",
        "/root/b/c.py",
        "/root/b/e.py",
        "/root/pkg_b/__init__.py",
        "/root/pkg_b/f.py",
        "/root/pkg_b/g/h.py",
        "/root/script.py",
    }


serialized_fn_path = "pkg_a/serialized_fn.py"


def test_mounted_files_serialized(servicer, supports_dir, env_mount_files, server_url_env):
    helpers.deploy_stub_externally(servicer, serialized_fn_path, cwd=supports_dir)
    files = set(servicer.files_name2sha.keys()) - set(env_mount_files)

    # Assert we include everything from `pkg_a` and `pkg_b` but not `pkg_c`:
    assert files == {
        "/root/b/c.py",
        "/root/b/e.py",
        "/root/pkg_a/a.py",
        "/root/pkg_a/serialized_fn.py",
        "/root/pkg_b/__init__.py",
        "/root/pkg_b/f.py",
        "/root/pkg_b/g/h.py",
    }


def test_mounted_files_package(supports_dir, env_mount_files, servicer, server_url_env):
    p = subprocess.run(["modal", "run", "pkg_a.package"], cwd=supports_dir)
    assert p.returncode == 0

    files = set(servicer.files_name2sha.keys()) - set(env_mount_files)
    # Assert we include everything from `pkg_a` and `pkg_b` but not `pkg_c`:
    assert files == {
        "/root/pkg_a/__init__.py",
        "/root/pkg_a/a.py",
        "/root/pkg_a/b/c.py",
        "/root/pkg_a/d.py",
        "/root/pkg_a/b/e.py",
        "/root/pkg_a/script.py",
        "/root/pkg_a/serialized_fn.py",
        "/root/pkg_a/package.py",
        "/root/pkg_b/__init__.py",
        "/root/pkg_b/f.py",
        "/root/pkg_b/g/h.py",
    }


def test_mounted_files_package_no_automount(supports_dir, env_mount_files, servicer, server_url_env):
    # when triggered like a module, the target module should be put at the correct package path
    p = subprocess.run(
        ["modal", "run", "pkg_a.package"],
        cwd=supports_dir,
        capture_output=True,
        env={**os.environ, "MODAL_AUTOMOUNT": "0"},
    )
    assert p.returncode == 0
    files = set(servicer.files_name2sha.keys()) - set(env_mount_files)
    assert files == {
        "/root/pkg_a/__init__.py",
        "/root/pkg_a/package.py",
    }


@skip_windows("venvs behave differently on Windows.")
def test_mounted_files_sys_prefix(servicer, supports_dir, venv_path, env_mount_files, server_url_env):
    # Run with venv activated, so it's on sys.prefix, and modal is dev-installed in the VM
    subprocess.run(
        [venv_path / "bin" / "modal", "run", script_path],
        cwd=supports_dir,
    )
    files = set(servicer.files_name2sha.keys()) - set(env_mount_files)
    # Assert we include everything from `pkg_a` and `pkg_b` but not `pkg_c`:
    assert files == {
        "/root/a.py",
        "/root/b/c.py",
        "/root/b/e.py",
        "/root/script.py",
        "/root/pkg_b/__init__.py",
        "/root/pkg_b/f.py",
        "/root/pkg_b/g/h.py",
    }


def test_mounted_files_config(servicer, supports_dir, env_mount_files, server_url_env):
    p = subprocess.run(
        ["modal", "run", "pkg_a/script.py"], cwd=supports_dir, env={**os.environ, "MODAL_AUTOMOUNT": "0"}
    )
    assert p.returncode == 0
    files = set(servicer.files_name2sha.keys()) - set(env_mount_files)
    assert files == {
        "/root/script.py",
    }


def test_e2e_modal_run_py_file_mounts(servicer, test_dir):
    helpers.deploy_stub_externally(servicer, "hello.py", cwd=test_dir.parent / "modal_test_support")
    # Reactivate the following mount assertions when we remove auto-mounting of dev-installed packages
    # assert len(servicer.files_name2sha) == 1
    # assert servicer.n_mounts == 1  # there should be a single mount
    # assert servicer.n_mount_files == 1
    assert "/root/hello.py" in servicer.files_name2sha


def test_e2e_modal_run_py_module_mounts(servicer, test_dir):
    helpers.deploy_stub_externally(servicer, "hello", cwd=test_dir.parent / "modal_test_support")
    # Reactivate the following mount assertions when we remove auto-mounting of dev-installed packages
    # assert len(servicer.files_name2sha) == 1
    # assert servicer.n_mounts == 1  # there should be a single mount
    # assert servicer.n_mount_files == 1
    assert "/root/hello.py" in servicer.files_name2sha
