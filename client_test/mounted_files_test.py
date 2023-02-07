# Copyright Modal Labs 2022
import os
import platform
import pytest
import subprocess
import sys
from pathlib import Path


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


def test_mounted_files_script(supports_dir):
    p = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        cwd=supports_dir,
        env={**os.environ, "PYTHONPATH": str(supports_dir)},
    )

    stdout = p.stdout.decode("utf-8")
    stderr = p.stderr.decode("utf-8")
    print("stdout: ", stdout)
    print("stderr: ", stderr)
    assert p.returncode == 0
    files = set(stdout.splitlines())

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


def test_mounted_files_serialized(supports_dir):
    p = subprocess.run(
        [sys.executable, str(serialized_fn_path)],
        capture_output=True,
        cwd=supports_dir,
        env={**os.environ, "PYTHONPATH": str(supports_dir)},
    )
    assert p.returncode == 0
    stdout = p.stdout.decode("utf-8")
    stderr = p.stderr.decode("utf-8")
    print("stdout: ", stdout)
    print("stderr: ", stderr)
    files = set(stdout.splitlines())

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


def test_mounted_files_package(supports_dir):
    p = subprocess.run([sys.executable, "-m", "pkg_a.package"], cwd=supports_dir, capture_output=True)
    assert p.returncode == 0
    stdout = p.stdout.decode("utf-8")
    stderr = p.stderr.decode("utf-8")
    print("stdout: ", stdout)
    print("stderr: ", stderr)
    files = set(stdout.splitlines())

    # Assert we include everything from `pkg_a` and `pkg_b` but not `pkg_c`:
    assert files == {
        "/root/package.py",
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


@pytest.mark.skipif(platform.system() == "Windows", reason="venvs behave differently on Windows.")
def test_mounted_files_sys_prefix(supports_dir, venv_path):
    # Run with venv activated, so it's on sys.prefix, and modal is dev-installed in the VM
    p = subprocess.run(
        [venv_path / "bin" / "python", script_path],
        capture_output=True,
        cwd=supports_dir,
        env={**os.environ, "PYTHONPATH": str(supports_dir)},
    )
    assert p.returncode == 0
    stdout = p.stdout.decode("utf-8")
    stderr = p.stderr.decode("utf-8")
    print("stdout: ", stdout)
    print("stderr: ", stderr)
    files = set(stdout.splitlines())

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


def test_mounted_files_config(supports_dir):
    p = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        cwd=supports_dir,
        env={**os.environ, "PYTHONPATH": str(supports_dir), "MODAL_AUTOMOUNT": ""},
    )
    assert p.returncode == 0
    stdout = p.stdout.decode("utf-8")
    stderr = p.stderr.decode("utf-8")
    print("stdout: ", stdout)
    print("stderr: ", stderr)
    files = set(stdout.splitlines())

    # Assert just the script is there
    assert files == {"/root/script.py"}
