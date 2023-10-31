# Copyright Modal Labs 2023
import pytest
from unittest import mock

import modal
from modal.exception import InvalidError
from modal.runner import deploy_stub


def dummy():
    pass


def test_volume_mount(client, servicer):
    stub = modal.Stub()

    _ = stub.function(
        volumes={"/root/foo": modal.Volume.new()},
    )(dummy)

    with stub.run(client=client):
        pass


def test_volume_bad_paths():
    stub = modal.Stub()

    with pytest.raises(InvalidError):
        stub.function(volumes={"/root/../../foo": modal.Volume.new()})(dummy)

    with pytest.raises(InvalidError):
        stub.function(volumes={"/": modal.Volume.new()})(dummy)

    with pytest.raises(InvalidError):
        stub.function(volumes={"/tmp/": modal.Volume.new()})(dummy)


def test_volume_duplicate_mount():
    stub = modal.Stub()

    volume = modal.Volume.new()
    with pytest.raises(InvalidError):
        stub.function(volumes={"/foo": volume, "/bar": volume})(dummy)


def test_volume_commit(client, servicer):
    stub = modal.Stub()
    stub.vol = modal.Volume.new()

    with stub.run(client=client):
        # Note that in practice this will not work unless run in a task.
        stub.vol.commit()

        # Make sure we can commit through the provider too
        stub.vol.commit()

        assert servicer.volume_commits[stub.vol.object_id] == 2
        # commit should implicitly reload on successful commit
        assert servicer.volume_reloads[stub.vol.object_id] == 2


@pytest.mark.asyncio
async def test_volume_get(servicer, client, tmp_path):
    stub = modal.Stub()
    vol = modal.Volume.persisted("my-vol")
    stub.vol = vol
    await vol._deploy.aio("my-vol", client=client)
    assert await modal.Volume._exists.aio("my-vol", client=client)  # type: ignore
    vol = await modal.Volume.lookup.aio("my-vol", client=client)  # type: ignore

    file_contents = b"hello world"
    file_path = b"foo.txt"
    local_file_path = tmp_path / file_path.decode("utf-8")
    local_file_path.write_bytes(file_contents)
    await vol._add_local_file.aio(local_file_path, file_path.decode("utf-8"))

    data = b""
    for chunk in vol.read_file(file_path):
        data += chunk
    assert data == file_contents


def test_volume_reload(client, servicer):
    stub = modal.Stub()
    stub.vol = modal.Volume.new()

    with stub.run(client=client):
        # Note that in practice this will not work unless run in a task.
        stub.vol.reload()

        assert servicer.volume_reloads[stub.vol.object_id] == 1


def test_redeploy(servicer, client):
    stub = modal.Stub()
    stub.v1 = modal.Volume.new()
    stub.v2 = modal.Volume.new()
    stub.v3 = modal.Volume.new()

    # Deploy app once
    deploy_stub(stub, "my-app", client=client)
    app1_ids = [stub.v1.object_id, stub.v2.object_id, stub.v3.object_id]

    # Deploy app again
    deploy_stub(stub, "my-app", client=client)
    app2_ids = [stub.v1.object_id, stub.v2.object_id, stub.v3.object_id]

    # Make sure ids are stable
    assert app1_ids == app2_ids

    # Make sure ids are unique
    assert len(set(app1_ids)) == 3
    assert len(set(app2_ids)) == 3

    # Deploy to a different app
    deploy_stub(stub, "my-other-app", client=client)
    app3_ids = [stub.v1.object_id, stub.v2.object_id, stub.v3.object_id]

    # Should be unique and different
    assert len(set(app3_ids)) == 3
    assert set(app1_ids) & set(app3_ids) == set()


@pytest.mark.asyncio
async def test_volume_add_local_file(servicer, client, tmp_path):
    stub = modal.Stub()
    stub.vol = modal.Volume.new()
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    with stub.run(client=client):
        stub.vol._add_local_file(local_file_path)
        stub.vol._add_local_file(local_file_path.as_posix(), remote_path="/foo/other_destination")
        object_id = stub.vol.object_id

    assert servicer.volume_files[object_id].keys() == {
        "/some_file",
        "/foo/other_destination",
    }
    assert servicer.volume_files[object_id]["/some_file"].data == b"hello world"
    assert servicer.volume_files[object_id]["/foo/other_destination"].data == b"hello world"


@pytest.mark.asyncio
async def test_volume_add_local_dir(client, tmp_path, servicer):
    stub = modal.Stub()
    stub.vol = modal.Volume.new()
    local_dir = tmp_path / "some_dir"
    local_dir.mkdir()
    (local_dir / "smol").write_text("###")

    subdir = local_dir / "subdir"
    subdir.mkdir()
    (subdir / "other").write_text("####")

    with stub.run(client=client):
        stub.vol._add_local_dir(local_dir)
        object_id = stub.vol.object_id

    assert servicer.volume_files[object_id].keys() == {
        "/some_dir/smol",
        "/some_dir/subdir/other",
    }
    assert servicer.volume_files[object_id]["/some_dir/smol"].data == b"###"
    assert servicer.volume_files[object_id]["/some_dir/subdir/other"].data == b"####"


@pytest.mark.asyncio
async def test_volume_put_large_file(client, tmp_path, servicer, blob_server, *args):
    with mock.patch("modal._blob_utils.LARGE_FILE_LIMIT", 10):
        stub = modal.Stub()
        stub.vol = modal.Volume.new()
        local_file_path = tmp_path / "bigfile"
        local_file_path.write_text("hello world, this is a lot of text")

        async with stub.run(client=client):
            await stub.vol._add_local_file.aio(local_file_path)
            object_id = stub.vol.object_id

        assert servicer.volume_files[object_id].keys() == {"/bigfile"}
        assert servicer.volume_files[object_id]["/bigfile"].data == b""
        assert servicer.volume_files[object_id]["/bigfile"].data_blob_id == "bl-1"

        _, blobs = blob_server
        assert blobs["bl-1"] == b"hello world, this is a lot of text"
