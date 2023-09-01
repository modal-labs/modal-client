# Copyright Modal Labs 2022
import pytest
from unittest import mock

import modal
from modal.exception import DeprecationError, InvalidError
from modal.runner import deploy_stub

from .supports.skip import skip_windows


def dummy():
    pass


def test_network_file_system_files(client, test_dir, servicer):
    stub = modal.Stub()

    dummy_modal = stub.function(
        network_file_systems={"/root/foo": modal.NetworkFileSystem.new()},
    )(dummy)

    with stub.run(client=client):
        dummy_modal.remote()


@skip_windows("TODO: implement client-side path check on Windows.")
def test_network_file_system_bad_paths(client, test_dir, servicer):
    stub = modal.Stub()

    def _f():
        pass

    dummy_modal = stub.function(network_file_systems={"/root/../../foo": modal.NetworkFileSystem.new()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.remote()

    dummy_modal = stub.function(network_file_systems={"/": modal.NetworkFileSystem.new()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.remote()

    dummy_modal = stub.function(network_file_systems={"/tmp/": modal.NetworkFileSystem.new()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.remote()


def test_network_file_system_handle_single_file(client, tmp_path, servicer):
    stub = modal.Stub()
    stub.vol = modal.NetworkFileSystem.new()
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    with stub.run(client=client):
        stub.vol.add_local_file(local_file_path)
        stub.vol.add_local_file(local_file_path.as_posix(), remote_path="/foo/other_destination")
        object_id = stub.vol.object_id

    assert servicer.nfs_files[object_id].keys() == {
        "/some_file",
        "/foo/other_destination",
    }
    assert servicer.nfs_files[object_id]["/some_file"].data == b"hello world"
    assert servicer.nfs_files[object_id]["/foo/other_destination"].data == b"hello world"


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

    with stub.run(client=client):
        stub.vol.add_local_dir(local_dir)
        object_id = stub.vol.object_id

    assert servicer.nfs_files[object_id].keys() == {
        "/some_dir/smol",
        "/some_dir/subdir/other",
    }
    assert servicer.nfs_files[object_id]["/some_dir/smol"].data == b"###"
    assert servicer.nfs_files[object_id]["/some_dir/subdir/other"].data == b"####"


@pytest.mark.asyncio
async def test_network_file_system_handle_big_file(client, tmp_path, servicer, blob_server, *args):
    with mock.patch("modal.network_file_system.LARGE_FILE_LIMIT", 10):
        stub = modal.Stub()
        stub.vol = modal.NetworkFileSystem.new()
        local_file_path = tmp_path / "bigfile"
        local_file_path.write_text("hello world, this is a lot of text")

        async with stub.run(client=client):
            await stub.vol.add_local_file.aio(local_file_path)
            object_id = stub.vol.object_id

        assert servicer.nfs_files[object_id].keys() == {"/bigfile"}
        assert servicer.nfs_files[object_id]["/bigfile"].data == b""
        assert servicer.nfs_files[object_id]["/bigfile"].data_blob_id == "bl-1"

        _, blobs = blob_server
        assert blobs["bl-1"] == b"hello world, this is a lot of text"


def test_old_syntax(client, servicer):
    stub = modal.Stub()
    with pytest.raises(DeprecationError):
        stub.vol1 = modal.SharedVolume()
    with pytest.raises(DeprecationError):
        stub.vol2 = modal.SharedVolume.new()


def test_redeploy(servicer, client):
    stub = modal.Stub()
    stub.n1 = modal.NetworkFileSystem.new()
    stub.n2 = modal.NetworkFileSystem.new()
    stub.n3 = modal.NetworkFileSystem.new()

    # Deploy app once
    deploy_stub(stub, "my-app", client=client)
    app1_ids = [stub.n1.object_id, stub.n2.object_id, stub.n3.object_id]

    # Deploy app again
    deploy_stub(stub, "my-app", client=client)
    app2_ids = [stub.n1.object_id, stub.n2.object_id, stub.n3.object_id]

    # Make sure ids are stable
    assert app1_ids == app2_ids

    # Make sure ids are unique
    assert len(set(app1_ids)) == 3
    assert len(set(app2_ids)) == 3

    # Deploy to a different app
    deploy_stub(stub, "my-other-app", client=client)
    app3_ids = [stub.n1.object_id, stub.n2.object_id, stub.n3.object_id]

    # Should be unique and different
    assert len(set(app3_ids)) == 3
    assert set(app1_ids) & set(app3_ids) == set()


def test_write_file(client, tmp_path, servicer):
    stub = modal.Stub()
    stub.vol = modal.NetworkFileSystem.new()
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    with stub.run(client=client):
        stub.vol.write_file("remote_path.txt", open(local_file_path, "rb"))

        # Make sure we can write through the provider too
        stub.vol.write_file("remote_path.txt", open(local_file_path, "rb"))
