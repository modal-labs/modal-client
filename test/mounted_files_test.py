# Copyright Modal Labs 2022
import os
import pytest
import subprocess
from pathlib import Path, PurePosixPath

import modal
from modal.file_pattern_matcher import FilePatternMatcher, _AbstractPatternMatcher
from modal.mount import Mount, _MountDir

from . import helpers


@pytest.fixture
def path_with_symlinked_files(tmp_path):
    src = tmp_path / "foo.txt"
    src.write_text("Hello")
    trg = tmp_path / "bar.txt"
    trg.symlink_to(src)
    return tmp_path, {src, trg}


script_path = "pkg_a/script.py"


def f():
    pass


serialized_fn_path = "pkg_a/serialized_fn.py"


def serialized_function_no_automount(servicer, credentials, supports_dir, server_url_env):
    helpers.deploy_app_externally(servicer, credentials, serialized_fn_path, cwd=supports_dir)
    files = set(servicer.files_name2sha.keys())
    # We don't automount anymore, nothing should be mounted
    assert files == {}


def test_mounted_files_package(supports_dir, servicer, server_url_env, token_env):
    p = subprocess.run(["modal", "run", "pkg_a.package"], cwd=supports_dir)
    assert p.returncode == 0

    files = set(servicer.files_name2sha.keys())
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
    }


def test_mounted_files_config(servicer, supports_dir, server_url_env, token_env):
    p = subprocess.run(["modal", "run", "pkg_a/script.py"], cwd=supports_dir, env={**os.environ})
    assert p.returncode == 0
    files = set(servicer.files_name2sha.keys())
    assert files == {
        "/root/script.py",
    }


def test_e2e_modal_run_py_file_mounts(servicer, credentials, supports_dir):
    helpers.deploy_app_externally(servicer, credentials, "hello.py", cwd=supports_dir)
    # Reactivate the following mount assertions when we remove auto-mounting of dev-installed packages
    # assert len(servicer.files_name2sha) == 1
    # assert servicer.n_mounts == 1  # there should be a single mount
    # assert servicer.n_mount_files == 1
    assert "/root/hello.py" in servicer.files_name2sha


def test_e2e_modal_run_py_module_mounts(servicer, credentials, supports_dir):
    helpers.deploy_app_externally(servicer, credentials, "hello", cwd=supports_dir)
    # Reactivate the following mount assertions when we remove auto-mounting of dev-installed packages
    # assert len(servicer.files_name2sha) == 1
    # assert servicer.n_mounts == 1  # there should be a single mount
    # assert servicer.n_mount_files == 1
    assert "/root/hello.py" in servicer.files_name2sha


def foo():
    pass


def test_image_mounts_are_not_traversed_on_declaration(supports_dir, monkeypatch, client, server_url_env):
    return_values = []
    original = modal.mount._MountDir.get_files_to_upload

    def mock_get_files_to_upload(self):
        r = list(original(self))
        return_values.append(r)
        return r

    monkeypatch.setattr("modal.mount._MountDir.get_files_to_upload", mock_get_files_to_upload)
    app = modal.App()
    image_mount_with_many_files = modal.Image.debian_slim().add_local_dir(supports_dir / "pkg_a", remote_path="/test")
    app.function(image=image_mount_with_many_files)(foo)
    assert len(return_values) == 0  # ensure we don't look at the files yet

    with app.run(client=client):
        pass

    assert return_values  # at this point we should have gotten all the mount files
    # flatten inspected files
    files = set()
    for r in return_values:
        for fn, _ in r:
            files.add(fn)
    # sanity check - this test file should be included since we mounted the test dir
    assert Path(__file__) in files  # this test file should have been included


def test_get_files_to_upload_ignore(mock_dir):
    with mock_dir(
        {
            "venv": {"file_venv": ""},
            "dir_a": {"dir_a_a": {"file_a_a": ""}, "venv": {"file_a_a_venv": ""}},
            "dir_b": {"dir_b_a": {"file_b_a": ""}, "venv": {"file_b_a_venv": ""}},
            "dir_c": {"venv": ""},  # a file named venv
        }
    ) as mock_dir:
        mock_path = Path(mock_dir).resolve()

        mount_dir = _MountDir(
            local_dir=mock_path,
            remote_path=PurePosixPath("/root"),
            ignore=FilePatternMatcher("**/venv/"),
            recursive=True,
        )

        # _walk_and_prune should prune out all ignored directories, but not yet files
        included_files = set(mount_dir._walk_and_prune(mock_path))
        assert len(included_files) == 3
        expected_files = {
            str(mock_path / "dir_a" / "dir_a_a" / "file_a_a"),
            str(mock_path / "dir_b" / "dir_b_a" / "file_b_a"),
            str(mock_path / "dir_c" / "venv"),  # a file named venv, not ignored
        }
        assert included_files == expected_files

        # after get_files_to_upload, both files and directories should be pruned out
        files = list(mount_dir.get_files_to_upload())
        assert len(files) == 2
        included_files = {str(file[0]) for file in files}
        expected_files = {
            str(mock_path / "dir_a" / "dir_a_a" / "file_a_a"),
            str(mock_path / "dir_b" / "dir_b_a" / "file_b_a"),
        }
        assert included_files == expected_files


@pytest.mark.parametrize(
    "ignore_config, expected_can_prune, expected_included",
    [
        (
            FilePatternMatcher("venv/**"),
            True,
            {"toplevel.py", "toplevel.pyc"},
        ),
        (
            FilePatternMatcher("**", "!**/*.py"),
            False,
            {"toplevel.py", "lib.py"},
        ),
        (
            ~FilePatternMatcher("**/*.py"),
            False,
            {"toplevel.py", "lib.py"},
        ),
    ],
)
def test_directory_pruning_behavior(mock_dir, ignore_config, expected_can_prune, expected_included):
    """Test that directory pruning is conditionally applied based on can_prune_directories()"""
    with mock_dir(
        {
            "toplevel.py": "",
            "toplevel.pyc": "",
            "venv": {
                "lib.py": "",
                "lib.pyc": "",
            },
        }
    ) as mock_dir:
        mock_path = Path(mock_dir).resolve()

        mount_dir = _MountDir(
            local_dir=mock_path,
            remote_path=PurePosixPath("/root"),
            ignore=ignore_config,
            recursive=True,
        )

        assert isinstance(mount_dir.ignore, _AbstractPatternMatcher)
        assert mount_dir.ignore.can_prune_directories() is expected_can_prune
        files = list(mount_dir.get_files_to_upload())
        file_paths = {f[0].name for f in files}

        assert file_paths == expected_included


def test_mount_dedupe_explicit(servicer, credentials, supports_dir, server_url_env):
    normally_not_included_file = supports_dir / "pkg_a" / "normally_not_included.pyc"
    normally_not_included_file.touch(exist_ok=True)
    print(
        helpers.deploy_app_externally(
            # two explicit mounts of the same package
            servicer,
            credentials,
            "mount_dedupe.py",
            cwd=supports_dir,
        )
    )
    assert servicer.n_mounts == 3

    # mounts are loaded in parallel, but there
    mounted_files_sets = {frozenset(m.keys()) for m in servicer.mounts_excluding_published_client().values()}
    assert {"/root/mount_dedupe.py"} in mounted_files_sets
    mounted_files_sets.remove(frozenset({"/root/mount_dedupe.py"}))

    # find one mount that includes normally_not_included.py
    for mount_with_pyc in mounted_files_sets:
        if "/root/pkg_a/normally_not_included.pyc" in mount_with_pyc:
            break
    else:
        assert False, "could not find a mount with normally_not_included.pyc"
    mounted_files_sets.remove(mount_with_pyc)

    # and one without it
    remaining_mount = list(mounted_files_sets)[0]
    assert "/root/pkg_a/normally_not_included.pyc" not in remaining_mount
    for fn in remaining_mount:
        assert fn.startswith("/root/pkg_a")

    assert len(mount_with_pyc) == len(remaining_mount) + 1
    normally_not_included_file.unlink()  # cleanup


def test_mount_dedupe_relative_path_entrypoint(servicer, credentials, supports_dir, server_url_env, monkeypatch):
    workdir = supports_dir / "pkg_a"
    target_app = "../hello.py"  # in parent directory - requiring `..` expansion in path normalization

    helpers.deploy_app_externally(
        # two explicit mounts of the same package
        servicer,
        credentials,
        target_app,
        cwd=workdir,
    )
    # should be only one unique set of files in mounts
    mounted_files_sets = {frozenset(m.keys()) for m in servicer.mounts_excluding_published_client().values()}
    assert len(mounted_files_sets) == 1

    # but there should also be only one actual mount if deduplication works as expected
    assert len(servicer.mounts_excluding_published_client()) == 1


def test_mount_directory_with_symlinked_file(path_with_symlinked_files, servicer, client):
    path, files = path_with_symlinked_files
    mount = Mount._from_local_dir(path)
    mount._deploy("mo-1", client=client)
    pkg_a_mount = servicer.mount_contents["mo-1"]
    for src_f in files:
        assert any(mnt_f.endswith(src_f.name) for mnt_f in pkg_a_mount)


def test_module_with_dot_prefixed_parent_can_be_mounted(tmp_path, monkeypatch, servicer, client):
    # the typical usecase would be to have a `.venv` directory with a virualenv
    # that could possibly contain local site-packages that a user wants to mount

    # set up some dummy packages:
    # .parent
    #    |---- foo.py
    #    |---- bar
    #    |------|--baz.py
    #    |------|--.hidden_dir
    #    |------|------|-----mod.py
    #    |------|--.hidden_mod.py

    parent_dir = Path(tmp_path) / ".parent"
    parent_dir.mkdir()
    foo_py = parent_dir / "foo.py"
    foo_py.touch()
    bar_package = parent_dir / "bar"
    bar_package.mkdir()
    (bar_package / "__init__.py").touch()
    (bar_package / "baz.py").touch()
    (bar_package / ".hidden_dir").mkdir()
    (bar_package / ".hidden_dir" / "mod.py").touch()  # should be excluded
    (bar_package / ".hidden_mod.py").touch()  # should be excluded

    monkeypatch.syspath_prepend(parent_dir)
    foo_mount = Mount._from_local_python_packages("foo")
    foo_mount._deploy("mo-1", client=client)
    foo_mount_content = servicer.mount_contents["mo-1"]
    assert foo_mount_content.keys() == {"/root/foo.py"}

    bar_mount = Mount._from_local_python_packages("bar")
    bar_mount._deploy("mo-2", client=client)

    bar_mount_content = servicer.mount_contents["mo-2"]
    assert bar_mount_content.keys() == {"/root/bar/__init__.py", "/root/bar/baz.py"}
