# Copyright Modal Labs 2022
import os
import pytest
import subprocess
import sys
from pathlib import Path

import pytest_asyncio

import modal
from modal import Mount
from modal._utils.function_utils import FunctionInfo

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


def f():
    pass


@pytest_asyncio.fixture
async def env_mount_files():
    # If something is installed using pip -e, it will be bundled up as a part of the environment.
    # Those are env-specific so we ignore those as a part of the test
    fn_info = FunctionInfo(f)

    filenames = []
    for mount in fn_info.get_auto_mounts():
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
    assert (
        files
        == {
            "/root/serialized_fn.py",  # should serialized_fn be included? It's not needed to run the function, but it's loaded into sys.modules at definition time...
            "/root/b/c.py",  # this is mounted under root since it's imported as `import b` and not `import pkg_a.b` from serialized_fn.py
            "/root/b/e.py",  # same as above
            "/root/a.py",  # same as above
            "/root/pkg_b/__init__.py",
            "/root/pkg_b/f.py",
            "/root/pkg_b/g/h.py",
        }
    )


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


def foo():
    pass


def test_mounts_are_not_traversed_on_declaration(test_dir, monkeypatch, client, server_url_env):
    return_values = []
    original = modal.mount._MountDir.get_files_to_upload

    def mock_get_files_to_upload(self):
        r = list(original(self))
        return_values.append(r)
        return r

    monkeypatch.setattr("modal.mount._MountDir.get_files_to_upload", mock_get_files_to_upload)
    stub = modal.Stub()
    mount_with_many_files = Mount.from_local_dir(test_dir, remote_path="/test")
    stub.function(mounts=[mount_with_many_files])(foo)
    assert len(return_values) == 0  # ensure we don't look at the files yet

    with stub.run(client=client):
        pass

    assert return_values  # at this point we should have gotten all the mount files
    # flatten inspected files
    files = set()
    for r in return_values:
        for fn, _ in r:
            files.add(fn)
    # sanity check - this test file should be included since we mounted the test dir
    assert __file__ in files  # this test file should have been included


def test_mount_dedupe(servicer, test_dir, server_url_env):
    supports_dir = test_dir / "supports"
    normally_not_included_file = supports_dir / "pkg_a" / "normally_not_included.pyc"
    normally_not_included_file.touch(exist_ok=True)
    print(
        helpers.deploy_stub_externally(
            # no explicit mounts, rely on auto-mounting
            servicer,
            "mount_dedupe.py",
            cwd=test_dir / "supports",
            env={"USE_EXPLICIT": "0"},
        )
    )
    assert servicer.n_mounts == 2
    assert servicer.mount_contents["mo-1"].keys() == {"/root/mount_dedupe.py"}
    pkg_a_mount = servicer.mount_contents["mo-2"]
    for fn in pkg_a_mount.keys():
        assert fn.startswith("/root/pkg_a")
    assert "/root/pkg_a/normally_not_included.pyc" not in pkg_a_mount.keys()


def test_mount_dedupe_explicit(servicer, test_dir, server_url_env):
    supports_dir = test_dir / "supports"
    normally_not_included_file = supports_dir / "pkg_a" / "normally_not_included.pyc"
    normally_not_included_file.touch(exist_ok=True)
    print(
        helpers.deploy_stub_externally(
            # two explicit mounts of the same package
            servicer,
            "mount_dedupe.py",
            cwd=supports_dir,
            env={"USE_EXPLICIT": "1"},
        )
    )
    assert servicer.n_mounts == 3
    assert servicer.mount_contents["mo-1"].keys() == {"/root/mount_dedupe.py"}
    pkg_a_mount = servicer.mount_contents["mo-2"]
    for fn in pkg_a_mount.keys():
        assert fn.startswith("/root/pkg_a")
    assert "/root/pkg_a/normally_not_included.pyc" not in pkg_a_mount.keys()

    custom_pkg_a_mount = servicer.mount_contents["mo-3"]
    assert len(custom_pkg_a_mount) == len(pkg_a_mount) + 1
    assert "/root/pkg_a/normally_not_included.pyc" in custom_pkg_a_mount.keys()


@skip_windows("pip-installed pdm seems somewhat broken on windows")
def test_pdm_cache_automount_exclude(tmp_path, monkeypatch, supports_dir, servicer, server_url_env):
    # check that `pdm`'s cached packages are not included in automounts
    project_dir = Path(__file__).parent.parent
    monkeypatch.chdir(tmp_path)
    subprocess.run(["pdm", "init", "-n"], check=True)
    subprocess.run(
        ["pdm", "add", "--dev", project_dir], check=True
    )  # install workdir modal into venv, not using cache...
    subprocess.run(["pdm", "config", "--local", "install.cache", "on"], check=True)
    subprocess.run(["pdm", "add", "six"], check=True)  # single file module
    subprocess.run(
        ["pdm", "run", "modal", "deploy", supports_dir / "imports_six.py"], check=True
    )  # deploy a basically empty function

    files = set(servicer.files_name2sha.keys())
    assert files == {
        "/root/imports_six.py",
    }
