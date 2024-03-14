# Copyright Modal Labs 2023
import io
import pytest
from pathlib import Path
from unittest import mock

import modal
from modal.exception import InvalidError, NotFoundError, VolumeUploadTimeoutError
from modal.runner import deploy_stub
from modal_proto import api_pb2


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


@pytest.mark.parametrize("skip_reload", [False, True])
def test_volume_commit(client, servicer, skip_reload):
    stub = modal.Stub()
    stub.vol = modal.Volume.new()

    with servicer.intercept() as ctx:
        ctx.add_response("VolumeCommit", api_pb2.VolumeCommitResponse(skip_reload=skip_reload))
        ctx.add_response("VolumeCommit", api_pb2.VolumeCommitResponse(skip_reload=skip_reload))

        with stub.run(client=client):
            # Note that in practice this will not work unless run in a task.
            stub.vol.commit()

            # Make sure we can commit through the provider too
            stub.vol.commit()

            assert ctx.pop_request("VolumeCommit").volume_id == stub.vol.object_id
            assert ctx.pop_request("VolumeCommit").volume_id == stub.vol.object_id

            # commit should implicitly reload on successful commit if skip_reload=False
            assert servicer.volume_reloads[stub.vol.object_id] == 0 if skip_reload else 2


@pytest.mark.asyncio
async def test_volume_get(servicer, client, tmp_path):
    await modal.Volume.create_deployed.aio("my-vol", client=client)
    vol = await modal.Volume.lookup.aio("my-vol", client=client)  # type: ignore

    file_contents = b"hello world"
    file_path = b"foo.txt"
    local_file_path = tmp_path / file_path.decode("utf-8")
    local_file_path.write_bytes(file_contents)

    async with vol.batch_upload() as batch:
        batch.put_file(local_file_path, file_path.decode("utf-8"))

    data = b""
    for chunk in vol.read_file(file_path):
        data += chunk
    assert data == file_contents

    output = io.BytesIO()
    vol.read_file_into_fileobj(file_path, output)
    assert output.getvalue() == file_contents

    with pytest.raises(FileNotFoundError):
        for _ in vol.read_file("/abc/def/i-dont-exist-at-all"):
            ...


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
async def test_volume_batch_upload(servicer, client, tmp_path):
    stub = modal.Stub()
    stub.vol = modal.Volume.new()

    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    local_dir = tmp_path / "some_dir"
    local_dir.mkdir()
    (local_dir / "smol").write_text("###")

    subdir = local_dir / "subdir"
    subdir.mkdir()
    (subdir / "other").write_text("####")

    with stub.run(client=client):
        with open(local_file_path, "rb") as fp:
            with stub.vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/some_file")
                batch.put_directory(local_dir, "/some_dir")
                batch.put_file(io.BytesIO(b"data from a file-like object"), "/filelike", mode=0o600)
                batch.put_directory(local_dir, "/non-recursive", recursive=False)
                batch.put_file(fp, "/filelike2")
        object_id = stub.vol.object_id

    assert servicer.volume_files[object_id].keys() == {
        "/some_file",
        "/some_dir/smol",
        "/some_dir/subdir/other",
        "/filelike",
        "/non-recursive/smol",
        "/filelike2",
    }
    assert servicer.volume_files[object_id]["/some_file"].data == b"hello world"
    assert servicer.volume_files[object_id]["/some_dir/smol"].data == b"###"
    assert servicer.volume_files[object_id]["/some_dir/subdir/other"].data == b"####"
    assert servicer.volume_files[object_id]["/filelike"].data == b"data from a file-like object"
    assert servicer.volume_files[object_id]["/filelike"].mode == 0o600
    assert servicer.volume_files[object_id]["/non-recursive/smol"].data == b"###"
    assert servicer.volume_files[object_id]["/filelike2"].data == b"hello world"
    assert servicer.volume_files[object_id]["/filelike2"].mode == 0o644


@pytest.mark.asyncio
async def test_volume_batch_upload_force(servicer, client, tmp_path):
    stub = modal.Stub()
    stub.vol = modal.Volume.new()

    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    local_file_path2 = tmp_path / "some_file2"
    local_file_path2.write_text("overwritten")

    with stub.run(client=client):
        with servicer.intercept() as ctx:
            # Seed the volume
            with stub.vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/some_file")
            assert ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

            # Attempting to overwrite the file with force=False should result in an error
            with pytest.raises(FileExistsError):
                with stub.vol.batch_upload(force=False) as batch:
                    batch.put_file(local_file_path, "/some_file")
            assert ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files
            assert servicer.volume_files[stub.vol.object_id]["/some_file"].data == b"hello world"

            # Overwriting should work with force=True
            with stub.vol.batch_upload(force=True) as batch:
                batch.put_file(local_file_path2, "/some_file")
            assert not ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files
            assert servicer.volume_files[stub.vol.object_id]["/some_file"].data == b"overwritten"


@pytest.mark.asyncio
async def test_volume_upload_removed_file(servicer, client, tmp_path):
    stub = modal.Stub()
    stub.vol = modal.Volume.new()

    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    with stub.run(client=client):
        with pytest.raises(FileNotFoundError):
            with stub.vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/dest")
                local_file_path.unlink()


@pytest.mark.asyncio
async def test_volume_upload_large_file(client, tmp_path, servicer, blob_server, *args):
    with mock.patch("modal._utils.blob_utils.LARGE_FILE_LIMIT", 10):
        stub = modal.Stub()
        stub.vol = modal.Volume.new()
        local_file_path = tmp_path / "bigfile"
        local_file_path.write_text("hello world, this is a lot of text")

        async with stub.run(client=client):
            async with stub.vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/a")
            object_id = stub.vol.object_id

        assert servicer.volume_files[object_id].keys() == {"/a"}
        assert servicer.volume_files[object_id]["/a"].data == b""
        assert servicer.volume_files[object_id]["/a"].data_blob_id == "bl-1"

        _, blobs = blob_server
        assert blobs["bl-1"] == b"hello world, this is a lot of text"


@pytest.mark.asyncio
async def test_volume_upload_file_timeout(client, tmp_path, servicer, blob_server, *args):
    call_count = 0

    def mount_put_file(_request):
        nonlocal call_count
        call_count += 1
        return api_pb2.MountPutFileResponse(exists=False)

    with servicer.intercept() as ctx:
        ctx.override_default("MountPutFile", mount_put_file)
        with mock.patch("modal._utils.blob_utils.LARGE_FILE_LIMIT", 10):
            with mock.patch("modal.volume.VOLUME_PUT_FILE_CLIENT_TIMEOUT", 0.5):
                stub = modal.Stub()
                stub.vol = modal.Volume.new()
                local_file_path = tmp_path / "bigfile"
                local_file_path.write_text("hello world, this is a lot of text")

                async with stub.run(client=client):
                    with pytest.raises(VolumeUploadTimeoutError):
                        async with stub.vol.batch_upload() as batch:
                            batch.put_file(local_file_path, "/dest")

                assert call_count > 2


@pytest.mark.asyncio
async def test_volume_copy(client, tmp_path, servicer):
    # setup
    stub = modal.Stub()
    stub.vol = modal.Volume.new()

    ## test 1: copy src path to dst path ##
    src_path = "original.txt"
    dst_path = "copied.txt"
    local_file_path = tmp_path / src_path
    local_file_path.write_text("test copy")

    with stub.run(client=client):
        # add local file to volume
        async with stub.vol.batch_upload() as batch:
            batch.put_file(local_file_path, src_path)
        object_id = stub.vol.object_id

        # copy file from src_path to dst_path
        stub.vol.copy_files([src_path], dst_path)

    assert servicer.volume_files[object_id].keys() == {src_path, dst_path}

    assert servicer.volume_files[object_id][src_path].data == b"test copy"
    assert servicer.volume_files[object_id][dst_path].data == b"test copy"

    ## test 2: copy multiple files into a directory ##
    file_paths = ["file1.txt", "file2.txt"]

    with stub.run(client=client):
        for file_path in file_paths:
            local_file_path = tmp_path / file_path
            local_file_path.write_text("test copy")
            async with stub.vol.batch_upload() as batch:
                batch.put_file(local_file_path, file_path)
            object_id = stub.vol.object_id

        stub.vol.copy_files(file_paths, "test_dir")

    returned_volume_files = [Path(file) for file in servicer.volume_files[object_id].keys()]
    expected_volume_files = [
        Path(file) for file in ["file1.txt", "file2.txt", "test_dir/file1.txt", "test_dir/file2.txt"]
    ]

    assert returned_volume_files == expected_volume_files

    returned_file_data = {
        Path(entry): servicer.volume_files[object_id][entry] for entry in servicer.volume_files[object_id]
    }
    assert returned_file_data[Path("test_dir/file1.txt")].data == b"test copy"
    assert returned_file_data[Path("test_dir/file2.txt")].data == b"test copy"


def test_persisted(servicer, client):
    # Lookup should fail since it doesn't exist
    with pytest.raises(NotFoundError):
        modal.Volume.lookup("xyz", client=client)

    # Create it
    modal.Volume.lookup("xyz", create_if_missing=True, client=client)

    # Lookup should succeed now
    modal.Volume.lookup("xyz", client=client)
