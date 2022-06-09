import hashlib
import os
import pytest

from modal import App
from modal._blob_utils import LARGE_FILE_LIMIT
from modal.aio import AioApp
from modal.mount import AioMount, Mount


@pytest.mark.asyncio
async def test_get_files(servicer, client, tmpdir, mock_blob_upload_file):
    small_content = "# not much here"
    large_content = "a" * (LARGE_FILE_LIMIT + 1)

    tmpdir.join("small.py").write(small_content)
    tmpdir.join("large.py").write(large_content)
    tmpdir.join("fluff").write("hello")

    files = {}
    app = AioApp()
    async with app.run(client=client) as running_app:
        m = AioMount("/", local_dir=tmpdir, condition=lambda fn: fn.endswith(".py"), recursive=True)
        await running_app.load(m)
        async for upload_spec in m._get_files():
            files[upload_spec.rel_filename] = upload_spec

        assert "small.py" in files
        assert "large.py" in files
        assert "fluff" not in files
        assert files["small.py"].use_blob is False
        assert files["small.py"].content == small_content.encode("utf8")
        assert files["small.py"].sha256_hex == hashlib.sha256(small_content.encode("utf8")).hexdigest()

        assert files["large.py"].use_blob is True
        assert files["large.py"].content is None
        assert files["large.py"].sha256_hex == hashlib.sha256(large_content.encode("utf8")).hexdigest()
        assert len(mock_blob_upload_file) == 1
        assert mock_blob_upload_file["0"].endswith("large.py")

        assert servicer.files_sha2data[files["large.py"].sha256_hex] == {"data": b"", "data_blob_id": "0"}
        assert servicer.files_sha2data[files["small.py"].sha256_hex] == {
            "data": small_content.encode("utf8"),
            "data_blob_id": "",
        }


def test_create_mount(servicer, client):
    app = App()
    with app.run(client=client) as running_app:
        local_dir, cur_filename = os.path.split(__file__)
        remote_dir = "/foo"

        def condition(fn):
            return fn.endswith(".py")

        m = Mount(local_dir=local_dir, remote_dir=remote_dir, condition=condition)
        obj_id = running_app.load(m)
        assert obj_id == "mo-123"
        assert f"/foo/{cur_filename}" in servicer.files_name2sha
        sha256_hex = servicer.files_name2sha[f"/foo/{cur_filename}"]
        assert sha256_hex in servicer.files_sha2data
        assert servicer.files_sha2data[sha256_hex]["data"] == open(__file__, "rb").read()


def test_create_mount_file_errors(servicer, client):
    app = App()
    with app.run(client=client) as running_app:
        m = Mount(local_dir="xyz", remote_dir="/xyz")
        with pytest.raises(FileNotFoundError):
            running_app.load(m)

        with open("abc", "w"):
            pass
        m = Mount(local_dir="abc", remote_dir="/abc")
        with pytest.raises(NotADirectoryError):
            running_app.load(m)
