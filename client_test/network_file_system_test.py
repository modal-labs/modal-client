# Copyright Modal Labs 2022
import pytest
from unittest import mock

import modal
from modal.exception import DeprecationError, InvalidError
from modal.network_file_system import NetworkFileSystemHandle

from .supports.skip import skip_windows


def dummy():
    pass


def test_network_file_system_files(client, test_dir, servicer):
    stub = modal.Stub()

    dummy_modal = stub.function(
        network_file_systems={"/root/foo": modal.NetworkFileSystem.new()},
    )(dummy)

    with stub.run(client=client):
        dummy_modal.call()


@skip_windows("TODO: implement client-side path check on Windows.")
def test_network_file_system_bad_paths(client, test_dir, servicer):
    stub = modal.Stub()

    def _f():
        pass

    dummy_modal = stub.function(network_file_systems={"/root/../../foo": modal.NetworkFileSystem.new()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()

    dummy_modal = stub.function(network_file_systems={"/": modal.NetworkFileSystem.new()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()

    dummy_modal = stub.function(network_file_systems={"/tmp/": modal.NetworkFileSystem.new()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()


def test_network_file_system_handle_single_file(client, tmp_path, servicer):
    stub = modal.Stub()
    stub.vol = modal.NetworkFileSystem.new()
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    with stub.run(client=client) as app:
        handle = app.vol
        assert isinstance(handle, NetworkFileSystemHandle)
        handle.add_local_file(local_file_path)
        handle.add_local_file(local_file_path.as_posix(), remote_path="/foo/other_destination")

    assert servicer.shared_volume_files[handle.object_id].keys() == {
        "/some_file",
        "/foo/other_destination",
    }
    assert servicer.shared_volume_files[handle.object_id]["/some_file"].data == b"hello world"
    assert servicer.shared_volume_files[handle.object_id]["/foo/other_destination"].data == b"hello world"


@pytest.mark.asyncio
async def test_network_file_system_handle_dir(client, tmp_path, servicer):
    stub = modal.Stub()
    stub.vol = modal.NetworkFileSystem.new()
    local_dir = tmp_path / "some_dir"
    local_dir.mkdir()
    (local_dir / "smol").write_text("###")

    subdir = local_dir / "subdir"
    subdir.mkdir()
    (subdir / "other").write_text("####")

    with stub.run(client=client) as app:
        handle = app.vol
        assert isinstance(handle, NetworkFileSystemHandle)
        handle.add_local_dir(local_dir)

    assert servicer.shared_volume_files[handle.object_id].keys() == {
        "/some_dir/smol",
        "/some_dir/subdir/other",
    }
    assert servicer.shared_volume_files[handle.object_id]["/some_dir/smol"].data == b"###"
    assert servicer.shared_volume_files[handle.object_id]["/some_dir/subdir/other"].data == b"####"


@pytest.mark.asyncio
async def test_network_file_system_handle_big_file(client, tmp_path, servicer, blob_server, *args):
    with mock.patch("modal.network_file_system.LARGE_FILE_LIMIT", 10):
        stub = modal.Stub()
        stub.vol = modal.NetworkFileSystem.new()
        local_file_path = tmp_path / "bigfile"
        local_file_path.write_text("hello world, this is a lot of text")

        async with stub.run(client=client) as app:
            handle = app.vol
            assert isinstance(handle, NetworkFileSystemHandle)
            await handle.add_local_file.aio(local_file_path)

        assert servicer.shared_volume_files[handle.object_id].keys() == {"/bigfile"}
        assert servicer.shared_volume_files[handle.object_id]["/bigfile"].data == b""
        assert servicer.shared_volume_files[handle.object_id]["/bigfile"].data_blob_id == "bl-1"

        _, blobs = blob_server
        assert blobs["bl-1"] == b"hello world, this is a lot of text"


def test_old_syntax(client, servicer):
    stub = modal.Stub()
    with pytest.warns(DeprecationError):
        stub.vol1 = modal.SharedVolume()
    with pytest.warns(DeprecationError):
        stub.vol2 = modal.SharedVolume.new()
    stub.vol3 = modal.NetworkFileSystem.new()
    with stub.run(client=client) as app:
        assert isinstance(app.vol1, NetworkFileSystemHandle)
        assert isinstance(app.vol2, NetworkFileSystemHandle)
        assert isinstance(app.vol3, NetworkFileSystemHandle)
