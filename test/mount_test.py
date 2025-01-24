# Copyright Modal Labs 2022
import hashlib
import os
import platform
import pytest
from pathlib import Path, PurePosixPath
from test.helpers import deploy_app_externally

from modal import App, FilePatternMatcher
from modal._utils.blob_utils import LARGE_FILE_LIMIT
from modal.mount import Mount, module_mount_condition, module_mount_ignore_condition


@pytest.mark.asyncio
async def test_get_files(servicer, client, tmpdir):
    small_content = b"# not much here"
    large_content = b"a" * (LARGE_FILE_LIMIT + 1)

    tmpdir.join("small.py").write(small_content)
    tmpdir.join("large.py").write(large_content)
    tmpdir.join("fluff").write("hello")

    files = {}
    m = Mount._from_local_dir(Path(tmpdir), remote_path="/", condition=lambda fn: fn.endswith(".py"), recursive=True)
    async for upload_spec in Mount._get_files.aio(m.entries):
        files[upload_spec.mount_filename] = upload_spec

    os.umask(umask := os.umask(0o022))  # Get the current umask
    expected_mode = 0o644 if platform.system() == "Windows" else 0o666 - umask

    assert "/small.py" in files
    assert "/large.py" in files
    assert "/fluff" not in files
    assert files["/small.py"].use_blob is False
    assert files["/small.py"].content == small_content
    assert files["/small.py"].sha256_hex == hashlib.sha256(small_content).hexdigest()
    assert files["/small.py"].mode == expected_mode

    assert files["/large.py"].use_blob is True
    assert files["/large.py"].content is None
    assert files["/large.py"].sha256_hex == hashlib.sha256(large_content).hexdigest()
    assert files["/large.py"].mode == expected_mode

    await m._deploy.aio("my-mount", client=client)
    blob_id = max(servicer.blobs.keys())  # last uploaded one
    assert len(servicer.blobs[blob_id]) == len(large_content)
    assert servicer.blobs[blob_id] == large_content

    assert servicer.files_sha2data[files["/large.py"].sha256_hex] == {"data": b"", "data_blob_id": blob_id}
    assert servicer.files_sha2data[files["/small.py"].sha256_hex] == {
        "data": small_content,
        "data_blob_id": "",
    }


def test_create_mount(servicer, client):
    local_dir, cur_filename = os.path.split(__file__)

    def condition(fn):
        return fn.endswith(".py")

    m = Mount._from_local_dir(local_dir, remote_path="/foo", condition=condition)

    m._deploy("my-mount", client=client)

    assert m.object_id == "mo-1"
    assert f"/foo/{cur_filename}" in servicer.files_name2sha
    sha256_hex = servicer.files_name2sha[f"/foo/{cur_filename}"]
    assert sha256_hex in servicer.files_sha2data
    assert servicer.files_sha2data[sha256_hex]["data"] == open(__file__, "rb").read()
    assert repr(Path(local_dir)) in repr(m)


def test_create_mount_file_errors(servicer, tmpdir, client):
    m = Mount._from_local_dir(Path(tmpdir) / "xyz", remote_path="/xyz")
    with pytest.raises(FileNotFoundError):
        m._deploy("my-mount", client=client)

    with open(tmpdir / "abc", "w"):
        pass
    m = Mount._from_local_dir(Path(tmpdir) / "abc", remote_path="/abc")
    with pytest.raises(NotADirectoryError):
        m._deploy("my-mount", client=client)


def dummy():
    pass


def test_from_local_python_packages(servicer, client, test_dir, monkeypatch):
    app = App()

    monkeypatch.syspath_prepend((test_dir / "supports").as_posix())

    app.function(mounts=[Mount._from_local_python_packages("pkg_a", "pkg_b", "standalone_file")])(dummy)

    with app.run(client=client):
        files = set(servicer.files_name2sha.keys())
        expected_files = {
            "/root/pkg_a/a.py",
            "/root/pkg_a/b/c.py",
            "/root/pkg_b/f.py",
            "/root/pkg_b/g/h.py",
            "/root/standalone_file.py",
        }
        assert expected_files.issubset(files)

        assert "/root/pkg_c/i.py" not in files
        assert "/root/pkg_c/j/k.py" not in files


def test_chained_entries(test_dir):
    # TODO: remove when public Mount is deprecated
    a_txt = str(test_dir / "a.txt")
    b_txt = str(test_dir / "b.txt")
    with open(a_txt, "w") as f:
        f.write("A")
    with open(b_txt, "w") as f:
        f.write("B")
    mount = Mount._from_local_file(a_txt).add_local_file(b_txt)
    entries = mount.entries
    assert len(entries) == 2
    files = [file for file in Mount._get_files(entries)]
    assert len(files) == 2
    files.sort(key=lambda file: file.source_description)
    assert files[0].source_description.name == "a.txt"
    assert files[0].mount_filename.endswith("/a.txt")
    assert files[0].content == b"A"
    m = hashlib.sha256()
    m.update(b"A")
    assert files[0].sha256_hex == m.hexdigest()
    assert files[0].use_blob is False


def test_module_mount_condition():
    condition = module_mount_condition(Path("/a/.venv/site-packages/mymod"))
    ignore_condition = module_mount_ignore_condition(Path("/a/.venv/site-packages/mymod"))

    include_paths = [
        Path("/a/.venv/site-packages/mymod/foo.py"),
        Path("/a/my_mod/config/foo.txt"),
        Path("/a/my_mod/config/foo.py"),
    ]
    exclude_paths = [
        Path("/a/site-packages/mymod/foo.pyc"),
        Path("/a/site-packages/mymod/__pycache__/foo.py"),
        Path("/a/my_mod/.config/foo.py"),
    ]
    for path in include_paths:
        assert condition(path)
        assert not ignore_condition(path)
    for path in exclude_paths:
        assert not condition(path)
        assert ignore_condition(path)


def test_mount_from_local_dir_ignore(test_dir, tmp_path_with_content):
    ignore = FilePatternMatcher("**/*.txt", "**/module", "!**/*.txt", "!**/*.py")
    expected = {
        "/foo/module/sub.py",
        "/foo/module/sub/sub.py",
        "/foo/data/sub",
        "/foo/module/__init__.py",
        "/foo/data.txt",
        "/foo/module/sub/__init__.py",
    }

    mount = Mount._add_local_dir(tmp_path_with_content, PurePosixPath("/foo"), ignore=ignore)

    file_names = [file.mount_filename for file in Mount._get_files(entries=mount.entries)]
    assert set(file_names) == expected


def test_missing_python_source_warning(servicer, credentials, supports_dir):
    # should warn if function doesn't have an imported non-third-party package attached
    # either through add OR copy mode, unless automount=False mode is used
    def has_warning(output: str):
        return 'image.add_local_python_source("pkg_a")' in output

    output = deploy_app_externally(servicer, credentials, "pkg_d.main", cwd=supports_dir, capture_output=True)
    assert has_warning(output)

    # adding the source to the image should make the warning disappear
    output = deploy_app_externally(
        servicer, credentials, "pkg_d.main", cwd=supports_dir, capture_output=True, env={"ADD_SOURCE": "add"}
    )
    assert not has_warning(output)

    # *copying* the source to the image should make the warning disappear too
    output = deploy_app_externally(
        servicer, credentials, "pkg_d.main", cwd=supports_dir, capture_output=True, env={"ADD_SOURCE": "copy"}
    )
    assert not has_warning(output)

    # disabling auto-mount explicitly should make warning disappear
    output = deploy_app_externally(
        servicer, credentials, "pkg_d.main", cwd=supports_dir, capture_output=True, env={"MODAL_AUTOMOUNT": "0"}
    )
    assert not has_warning(output)
