# Copyright Modal Labs 2022
from unittest import mock

import pytest

import modal
import modal.aio
from modal.exception import InvalidError
from modal.shared_volume import SharedVolumeHandle, AioSharedVolumeHandle

from .supports.skip import skip_windows


def dummy():
    pass


def test_shared_volume_files(client, test_dir, servicer):
    stub = modal.Stub()

    dummy_modal = stub.function(
        shared_volumes={"/root/foo": modal.SharedVolume()},
    )(dummy)

    with stub.run(client=client):
        dummy_modal.call()


@skip_windows("TODO: implement client-side path check on Windows.")
def test_shared_volume_bad_paths(client, test_dir, servicer):
    stub = modal.Stub()

    def _f():
        pass

    dummy_modal = stub.function(shared_volumes={"/root/../../foo": modal.SharedVolume()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()

    dummy_modal = stub.function(shared_volumes={"/": modal.SharedVolume()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()

    dummy_modal = stub.function(shared_volumes={"/tmp/": modal.SharedVolume()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()


def test_shared_volume_handle_single_file(client, tmp_path, servicer):
    stub = modal.Stub()
    stub.vol = modal.SharedVolume()
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    with stub.run(client=client) as app:
        handle = app.vol
        assert isinstance(handle, SharedVolumeHandle)
        handle.add_local_file(local_file_path)
        handle.add_local_file(local_file_path.as_posix(), remote_path="/foo/other_destination")

    assert servicer.shared_volume_files[handle.object_id].keys() == {
        "/some_file",
        "/foo/other_destination",
    }
    assert servicer.shared_volume_files[handle.object_id]["/some_file"].data == b"hello world"
    assert servicer.shared_volume_files[handle.object_id]["/foo/other_destination"].data == b"hello world"


@pytest.mark.asyncio
async def test_shared_volume_handle_dir(client, tmp_path, servicer):
    stub = modal.Stub()
    stub.vol = modal.SharedVolume()
    local_dir = tmp_path / "some_dir"
    local_dir.mkdir()
    (local_dir / "smol").write_text("###")

    subdir = local_dir / "subdir"
    subdir.mkdir()
    (subdir / "other").write_text("####")

    with stub.run(client=client) as app:
        handle = app.vol
        assert isinstance(handle, SharedVolumeHandle)
        handle.add_local_dir(local_dir)

    assert servicer.shared_volume_files[handle.object_id].keys() == {
        "/some_dir/smol",
        "/some_dir/subdir/other",
    }
    assert servicer.shared_volume_files[handle.object_id]["/some_dir/smol"].data == b"###"
    assert servicer.shared_volume_files[handle.object_id]["/some_dir/subdir/other"].data == b"####"


@pytest.mark.asyncio
async def test_shared_volume_handle_big_file(client, tmp_path, servicer, blob_server, *args):
    with mock.patch("modal.shared_volume.LARGE_FILE_LIMIT", 10):
        stub = modal.aio.AioStub()
        stub.vol = modal.aio.AioSharedVolume()
        local_file_path = tmp_path / "bigfile"
        local_file_path.write_text("hello world, this is a lot of text")

        async with stub.run(client=client) as app:
            handle = app.vol
            assert isinstance(handle, AioSharedVolumeHandle)
            await handle.add_local_file(local_file_path)

        assert servicer.shared_volume_files[handle.object_id].keys() == {"/bigfile"}
        assert servicer.shared_volume_files[handle.object_id]["/bigfile"].data == b""
        assert servicer.shared_volume_files[handle.object_id]["/bigfile"].data_blob_id == "bl-1"

        _, blobs = blob_server
        assert blobs["bl-1"] == b"hello world, this is a lot of text"
