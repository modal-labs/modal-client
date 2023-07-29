# Copyright Modal Labs 2022
import hashlib
import os
import pytest
import sys
from pathlib import Path

from modal import App, Stub
from modal._blob_utils import LARGE_FILE_LIMIT
from modal.exception import NotFoundError
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

    assert "/small.py" in files
    assert "/large.py" in files
    assert "/fluff" not in files
    assert files["/small.py"].use_blob is False
    assert files["/small.py"].content == small_content
    assert files["/small.py"].sha256_hex == hashlib.sha256(small_content).hexdigest()

    assert files["/large.py"].use_blob is True
    assert files["/large.py"].content is None
    assert files["/large.py"].sha256_hex == hashlib.sha256(large_content).hexdigest()

    app = await App._init_new.aio(client)
    await app.create_one_object.aio(m, "")
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

    app = App._init_new(client)
    obj = app.create_one_object(m, "")

    assert obj.object_id == "mo-123"
    assert f"/foo/{cur_filename}" in servicer.files_name2sha
    sha256_hex = servicer.files_name2sha[f"/foo/{cur_filename}"]
    assert sha256_hex in servicer.files_sha2data
    assert servicer.files_sha2data[sha256_hex]["data"] == open(__file__, "rb").read()
    assert repr(Path(local_dir)) in repr(m)


def test_create_mount_file_errors(servicer, tmpdir, client):
    app = App._init_new(client)
    m = Mount.from_local_dir(Path(tmpdir) / "xyz", remote_path="/xyz")
    with pytest.raises(FileNotFoundError):
        app.create_one_object(m, "")

    with open(tmpdir / "abc", "w"):
        pass
    m = Mount.from_local_dir(Path(tmpdir) / "abc", remote_path="/abc")
    with pytest.raises(NotADirectoryError):
        app.create_one_object(m, "")


def dummy():
    pass


def test_from_local_python_packages(servicer, client, test_dir):
    stub = Stub()

    sys.path.append((test_dir / "supports").as_posix())

    stub.function(mounts=[Mount.from_local_python_packages("pkg_a", "pkg_b", "standalone_file")])(dummy)

    with stub.run(client=client):
        files = servicer.files_name2sha.keys()
        assert any(["/pkg/pkg_a/a.py" in f for f in files])
        assert any(["/pkg/pkg_a/b/c.py" in f for f in files])
        assert any(["/pkg/pkg_b/f.py" in f for f in files])
        assert any(["/pkg/pkg_b/g/h.py" in f for f in files])
        assert any(["/pkg/standalone_file.py" in f for f in files])
        assert not any(["/pkg/pkg_c/i.py" in f for f in files])
        assert not any(["/pkg/pkg_c/j/k.py" in f for f in files])


def test_stub_mounts(servicer, client, test_dir):
    sys.path.append((test_dir / "supports").as_posix())

    stub = Stub(mounts=[Mount.from_local_python_packages("pkg_b")])

    stub.function(mounts=[Mount.from_local_python_packages("pkg_a")])(dummy)

    with stub.run(client=client):
        files = servicer.files_name2sha.keys()
        assert any(["pkg/pkg_a/a.py" in f for f in files])
        assert any(["pkg/pkg_a/b/c.py" in f for f in files])
        assert any(["pkg/pkg_b/f.py" in f for f in files])
        assert any(["pkg/pkg_b/g/h.py" in f for f in files])
        assert not any(["pkg/pkg_c/i.py" in f for f in files])
        assert not any(["pkg/pkg_c/j/k.py" in f for f in files])


def test_from_local_python_packages_missing_module(servicer, client, test_dir):
    stub = Stub()

    with pytest.raises(NotFoundError):
        stub.function(mounts=[Mount.from_local_python_packages("nonexistent_package")])(dummy)


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
    files.sort(key=lambda file: file.filename)
    assert files[0].filename.name == "a.txt"
    assert files[0].mount_filename.endswith("/a.txt")
    assert files[0].content == b"A"
    m = hashlib.sha256()
    m.update(b"A")
    assert files[0].sha256_hex == m.hexdigest()
    assert files[0].use_blob is False
