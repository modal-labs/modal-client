# Copyright Modal Labs 2022
import hashlib
import os
import platform
import pytest
import sys
from pathlib import Path

from modal import App
from modal._utils.blob_utils import LARGE_FILE_LIMIT
from modal.exception import ModuleNotMountable
from modal.mount import Mount


@pytest.mark.asyncio
async def test_get_files(servicer, client, tmpdir):
    small_content = b"# not much here"
    large_content = b"a" * (LARGE_FILE_LIMIT + 1)

    tmpdir.join("small.py").write(small_content)
    tmpdir.join("large.py").write(large_content)
    tmpdir.join("fluff").write("hello")

    files = {}
    m = Mount.from_local_dir(Path(tmpdir), remote_path="/", condition=lambda fn: fn.endswith(".py"), recursive=True)
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

    m = Mount.from_local_dir(local_dir, remote_path="/foo", condition=condition)

    m._deploy("my-mount", client=client)

    assert m.object_id == "mo-1"
    assert f"/foo/{cur_filename}" in servicer.files_name2sha
    sha256_hex = servicer.files_name2sha[f"/foo/{cur_filename}"]
    assert sha256_hex in servicer.files_sha2data
    assert servicer.files_sha2data[sha256_hex]["data"] == open(__file__, "rb").read()
    assert repr(Path(local_dir)) in repr(m)


def test_create_mount_file_errors(servicer, tmpdir, client):
    m = Mount.from_local_dir(Path(tmpdir) / "xyz", remote_path="/xyz")
    with pytest.raises(FileNotFoundError):
        m._deploy("my-mount", client=client)

    with open(tmpdir / "abc", "w"):
        pass
    m = Mount.from_local_dir(Path(tmpdir) / "abc", remote_path="/abc")
    with pytest.raises(NotADirectoryError):
        m._deploy("my-mount", client=client)


def dummy():
    pass


def test_from_local_python_packages(servicer, client, test_dir):
    app = App()

    sys.path.append((test_dir / "supports").as_posix())

    app.function(mounts=[Mount.from_local_python_packages("pkg_a", "pkg_b", "standalone_file")])(dummy)

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


def test_app_mounts(servicer, client, test_dir):
    sys.path.append((test_dir / "supports").as_posix())

    app = App(mounts=[Mount.from_local_python_packages("pkg_b")])

    app.function(mounts=[Mount.from_local_python_packages("pkg_a")])(dummy)

    with app.run(client=client):
        files = set(servicer.files_name2sha.keys())
        expected_files = {
            "/root/pkg_a/a.py",
            "/root/pkg_a/b/c.py",
            "/root/pkg_b/f.py",
            "/root/pkg_b/g/h.py",
        }
        assert expected_files.issubset(files)

        assert "/root/pkg_c/i.py" not in files
        assert "/root/pkg_c/j/k.py" not in files


def test_from_local_python_packages_missing_module(servicer, client, test_dir, server_url_env):
    app = App()
    app.function(mounts=[Mount.from_local_python_packages("nonexistent_package")])(dummy)

    with pytest.raises(ModuleNotMountable):
        with app.run(client=client):
            pass


def test_chained_entries(test_dir):
    a_txt = str(test_dir / "a.txt")
    b_txt = str(test_dir / "b.txt")
    with open(a_txt, "w") as f:
        f.write("A")
    with open(b_txt, "w") as f:
        f.write("B")
    mount = Mount.from_local_file(a_txt).add_local_file(b_txt)
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
