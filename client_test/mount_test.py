# Copyright Modal Labs 2022
import hashlib
import os
from pathlib import Path

import pytest
import sys

from modal import App, Stub, create_package_mounts
from modal._blob_utils import LARGE_FILE_LIMIT
from modal.aio import AioApp
from modal.exception import DeprecationError, NotFoundError
from modal.mount import AioMount, Mount


@pytest.mark.asyncio
async def test_get_files(servicer, aio_client, tmpdir):
    small_content = b"# not much here"
    large_content = b"a" * (LARGE_FILE_LIMIT + 1)

    tmpdir.join("small.py").write(small_content)
    tmpdir.join("large.py").write(large_content)
    tmpdir.join("fluff").write("hello")

    files = {}
    m = AioMount.from_local_dir(tmpdir, remote_path="/", condition=lambda fn: fn.endswith(".py"), recursive=True)
    async for upload_spec in m._get_files():
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

    await AioApp._create_one_object(aio_client, m)
    blob_id = max(servicer.blobs.keys())  # last uploaded one
    assert len(servicer.blobs[blob_id]) == len(large_content)
    assert servicer.blobs[blob_id] == large_content

    assert servicer.files_sha2data[files["/large.py"].sha256_hex] == {"data": b"", "data_blob_id": blob_id}
    assert servicer.files_sha2data[files["/small.py"].sha256_hex] == {
        "data": small_content,
        "data_blob_id": "",
    }


def test_create_mount_legacy_constructor(servicer, client):
    local_dir, cur_filename = os.path.split(__file__)
    remote_dir = "/foo"

    def condition(fn):
        return fn.endswith(".py")

    with pytest.warns(DeprecationError):
        m = Mount(local_dir=local_dir, remote_dir=remote_dir, condition=condition)

    obj, _ = App._create_one_object(client, m)

    assert obj.object_id == "mo-123"
    assert f"/foo/{cur_filename}" in servicer.files_name2sha
    sha256_hex = servicer.files_name2sha[f"/foo/{cur_filename}"]
    assert sha256_hex in servicer.files_sha2data
    assert servicer.files_sha2data[sha256_hex]["data"] == open(__file__, "rb").read()


def test_create_mount(servicer, client):
    local_dir, cur_filename = os.path.split(__file__)

    def condition(fn):
        return fn.endswith(".py")

    m = Mount.from_local_dir(local_dir, remote_path="/foo", condition=condition)

    obj, _ = App._create_one_object(client, m)

    assert obj.object_id == "mo-123"
    assert f"/foo/{cur_filename}" in servicer.files_name2sha
    sha256_hex = servicer.files_name2sha[f"/foo/{cur_filename}"]
    assert sha256_hex in servicer.files_sha2data
    assert servicer.files_sha2data[sha256_hex]["data"] == open(__file__, "rb").read()
    assert repr(Path(local_dir)) in repr(m)


def test_create_mount_file_errors(servicer, tmpdir, client):
    m = Mount.from_local_dir("xyz", remote_path="/xyz")
    with pytest.raises(FileNotFoundError):
        App._create_one_object(client, m)

    with open(tmpdir / "abc", "w"):
        pass
    m = Mount.from_local_dir(tmpdir / "abc", remote_path="/abc")
    with pytest.raises(NotADirectoryError):
        App._create_one_object(client, m)


def dummy():
    pass


def test_create_package_mounts(servicer, client, test_dir):
    stub = Stub()

    sys.path.append((test_dir / "supports").as_posix())

    stub.function(dummy, mounts=create_package_mounts(["pkg_a", "pkg_b", "standalone_file"]))

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

    stub = Stub(mounts=create_package_mounts(["pkg_b"]))

    stub.function(dummy, mounts=create_package_mounts(["pkg_a"]))

    with stub.run(client=client):
        files = servicer.files_name2sha.keys()
        assert any(["pkg/pkg_a/a.py" in f for f in files])
        assert any(["pkg/pkg_a/b/c.py" in f for f in files])
        assert any(["pkg/pkg_b/f.py" in f for f in files])
        assert any(["pkg/pkg_b/g/h.py" in f for f in files])
        assert not any(["pkg/pkg_c/i.py" in f for f in files])
        assert not any(["pkg/pkg_c/j/k.py" in f for f in files])


def test_create_package_mounts_missing_module(servicer, client, test_dir):
    stub = Stub()

    with pytest.raises(NotFoundError):
        stub.function(dummy, mounts=create_package_mounts(["nonexistent_package"]))
