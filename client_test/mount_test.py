# Copyright Modal Labs 2022
import hashlib
import os
import pytest
import sys

from modal import Stub, create_package_mounts
from modal._blob_utils import LARGE_FILE_LIMIT
from modal.aio import AioStub
from modal.exception import NotFoundError
from modal.mount import AioMount, Mount


@pytest.mark.asyncio
async def test_get_files(servicer, client, tmpdir):
    small_content = b"# not much here"
    large_content = b"a" * (LARGE_FILE_LIMIT + 1)

    tmpdir.join("small.py").write(small_content)
    tmpdir.join("large.py").write(large_content)
    tmpdir.join("fluff").write("hello")

    files = {}
    stub = AioStub()
    async with stub.run(client=client) as running_app:
        m = AioMount("/", local_dir=tmpdir, condition=lambda fn: fn.endswith(".py"), recursive=True)
        await running_app._load(m)  # TODO: is this something we want to expose?
        async for upload_spec in m._get_files():
            files[upload_spec.rel_filename] = upload_spec

        assert "small.py" in files
        assert "large.py" in files
        assert "fluff" not in files
        assert files["small.py"].use_blob is False
        assert files["small.py"].content == small_content
        assert files["small.py"].sha256_hex == hashlib.sha256(small_content).hexdigest()

        assert files["large.py"].use_blob is True
        assert files["large.py"].content is None
        assert files["large.py"].sha256_hex == hashlib.sha256(large_content).hexdigest()
        blob_id = max(servicer.blobs.keys())  # last uploaded one
        assert len(servicer.blobs[blob_id]) == len(large_content)
        assert servicer.blobs[blob_id] == large_content

        assert servicer.files_sha2data[files["large.py"].sha256_hex] == {"data": b"", "data_blob_id": blob_id}
        assert servicer.files_sha2data[files["small.py"].sha256_hex] == {
            "data": small_content,
            "data_blob_id": "",
        }


def test_create_mount(servicer, client):
    stub = Stub()
    with stub.run(client=client) as running_app:
        local_dir, cur_filename = os.path.split(__file__)
        remote_dir = "/foo"

        def condition(fn):
            return fn.endswith(".py")

        m = Mount(local_dir=local_dir, remote_dir=remote_dir, condition=condition)
        obj = running_app._load(m)  # TODO: is this something we want to expose?
        assert obj.object_id == "mo-123"
        assert f"/foo/{cur_filename}" in servicer.files_name2sha
        sha256_hex = servicer.files_name2sha[f"/foo/{cur_filename}"]
        assert sha256_hex in servicer.files_sha2data
        assert servicer.files_sha2data[sha256_hex]["data"] == open(__file__, "rb").read()


def test_create_mount_file_errors(servicer, tmpdir, client):
    stub = Stub()
    with stub.run(client=client) as running_app:
        m = Mount(local_dir="xyz", remote_dir="/xyz")
        with pytest.raises(FileNotFoundError):
            running_app._load(m)

        with open(tmpdir / "abc", "w"):
            pass
        m = Mount(local_dir=tmpdir / "abc", remote_dir="/abc")
        with pytest.raises(NotADirectoryError):
            running_app._load(m)


def test_create_package_mounts(servicer, client, test_dir):
    stub = Stub()

    sys.path.append((test_dir / "supports").as_posix())

    @stub.function(mounts=create_package_mounts(["pkg_a", "pkg_b", "standalone_file"]))
    def f():
        pass

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

    @stub.function(mounts=create_package_mounts(["pkg_a"]))
    def f():
        pass

    with stub.run(client=client):
        files = servicer.files_name2sha.keys()
        assert any(["/pkg/pkg_a/a.py" in f for f in files])
        assert any(["/pkg/pkg_a/b/c.py" in f for f in files])
        assert any(["/pkg/pkg_b/f.py" in f for f in files])
        assert any(["/pkg/pkg_b/g/h.py" in f for f in files])
        assert not any(["/pkg/pkg_c/i.py" in f for f in files])
        assert not any(["/pkg/pkg_c/j/k.py" in f for f in files])


def test_create_package_mounts_missing_module(servicer, client, test_dir):
    stub = Stub()

    with pytest.raises(NotFoundError):

        @stub.function(mounts=create_package_mounts(["nonexistent_package"]))
        def f():
            pass
