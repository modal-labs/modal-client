# Copyright Modal Labs 2023
import asyncio
import io
import os
import platform
import pytest
import re
import sys
import time
from pathlib import Path
from unittest import mock

import modal
from modal.exception import DeprecationError, InvalidError, NotFoundError, VolumeUploadTimeoutError
from modal.volume import _open_files_error_annotation
from modal_proto import api_pb2


def dummy():
    pass


def test_volume_mount(client, servicer):
    app = modal.App()
    vol = modal.Volume.from_name("xyz", create_if_missing=True)

    _ = app.function(volumes={"/root/foo": vol})(dummy)

    with app.run(client=client):
        pass


def test_volume_bad_paths():
    app = modal.App()
    vol = modal.Volume.from_name("xyz")

    with pytest.raises(InvalidError):
        app.function(volumes={"/root/../../foo": vol})(dummy)

    with pytest.raises(InvalidError):
        app.function(volumes={"/": vol})(dummy)

    with pytest.raises(InvalidError):
        app.function(volumes={"/tmp/": vol})(dummy)


def test_volume_duplicate_mount():
    app = modal.App()
    vol = modal.Volume.from_name("xyz")

    with pytest.raises(InvalidError):
        app.function(volumes={"/foo": vol, "/bar": vol})(dummy)


@pytest.mark.parametrize("skip_reload", [False, True])
def test_volume_commit(client, servicer, skip_reload):
    with servicer.intercept() as ctx:
        ctx.add_response("VolumeCommit", api_pb2.VolumeCommitResponse(skip_reload=skip_reload))
        ctx.add_response("VolumeCommit", api_pb2.VolumeCommitResponse(skip_reload=skip_reload))

        with modal.Volume.ephemeral(client=client) as vol:
            # Note that in practice this will not work unless run in a task.
            vol.commit()

            # Make sure we can commit through the provider too
            vol.commit()

            assert ctx.pop_request("VolumeCommit").volume_id == vol.object_id
            assert ctx.pop_request("VolumeCommit").volume_id == vol.object_id

            # commit should implicitly reload on successful commit if skip_reload=False
            assert servicer.volume_reloads[vol.object_id] == 0 if skip_reload else 2


@pytest.mark.asyncio
async def test_volume_get(servicer, client, tmp_path):
    await modal.Volume.create_deployed.aio("my-vol", client=client)
    vol = await modal.Volume.lookup.aio("my-vol", client=client)  # type: ignore

    file_contents = b"hello world"
    file_path = "foo.txt"
    local_file_path = tmp_path / file_path
    local_file_path.write_bytes(file_contents)

    async with vol.batch_upload() as batch:
        batch.put_file(local_file_path, file_path)

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
    with modal.Volume.ephemeral(client=client) as vol:
        # Note that in practice this will not work unless run in a task.
        vol.reload()

        assert servicer.volume_reloads[vol.object_id] == 1


@pytest.mark.asyncio
async def test_volume_batch_upload(servicer, client, tmp_path):
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    local_dir = tmp_path / "some_dir"
    local_dir.mkdir()
    (local_dir / "smol").write_text("###")

    subdir = local_dir / "subdir"
    subdir.mkdir()
    (subdir / "other").write_text("####")

    async with modal.Volume.ephemeral(client=client) as vol:
        with open(local_file_path, "rb") as fp:
            with vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/some_file")
                batch.put_directory(local_dir, "/some_dir")
                batch.put_file(io.BytesIO(b"data from a file-like object"), "/filelike", mode=0o600)
                batch.put_directory(local_dir, "/non-recursive", recursive=False)
                batch.put_file(fp, "/filelike2")
        object_id = vol.object_id

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
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    local_file_path2 = tmp_path / "some_file2"
    local_file_path2.write_text("overwritten")

    async with modal.Volume.ephemeral(client=client) as vol:
        with servicer.intercept() as ctx:
            # Seed the volume
            with vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/some_file")
            assert ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

            # Attempting to overwrite the file with force=False should result in an error
            with pytest.raises(FileExistsError):
                with vol.batch_upload(force=False) as batch:
                    batch.put_file(local_file_path, "/some_file")
            assert ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files
            assert servicer.volume_files[vol.object_id]["/some_file"].data == b"hello world"

            # Overwriting should work with force=True
            with vol.batch_upload(force=True) as batch:
                batch.put_file(local_file_path2, "/some_file")
            assert not ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files
            assert servicer.volume_files[vol.object_id]["/some_file"].data == b"overwritten"


@pytest.mark.asyncio
async def test_volume_upload_removed_file(servicer, client, tmp_path):
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    async with modal.Volume.ephemeral(client=client) as vol:
        with pytest.raises(FileNotFoundError):
            with vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/dest")
                local_file_path.unlink()


@pytest.mark.asyncio
async def test_volume_upload_large_file(client, tmp_path, servicer, blob_server, *args):
    with mock.patch("modal._utils.blob_utils.LARGE_FILE_LIMIT", 10):
        local_file_path = tmp_path / "bigfile"
        local_file_path.write_text("hello world, this is a lot of text")

        async with modal.Volume.ephemeral(client=client) as vol:
            async with vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/a")
            object_id = vol.object_id

        assert servicer.volume_files[object_id].keys() == {"/a"}
        assert servicer.volume_files[object_id]["/a"].data == b""
        assert servicer.volume_files[object_id]["/a"].data_blob_id == "bl-1"

        _, blobs = blob_server
        assert blobs["bl-1"] == b"hello world, this is a lot of text"

@pytest.mark.asyncio
async def test_volume_upload_large_stream(client, servicer, blob_server, *args):
    with mock.patch("modal._utils.blob_utils.LARGE_FILE_LIMIT", 10):
        stream = io.BytesIO(b"hello world, this is a lot of text")

        async with modal.Volume.ephemeral(client=client) as vol:
            async with vol.batch_upload() as batch:
                batch.put_file(stream, "/a")
            object_id = vol.object_id

        assert servicer.volume_files[object_id].keys() == {"/a"}
        assert servicer.volume_files[object_id]["/a"].data == b""
        assert servicer.volume_files[object_id]["/a"].data_blob_id == "bl-1"

        _, blobs = blob_server
        assert blobs["bl-1"] == b"hello world, this is a lot of text"

@pytest.mark.asyncio
async def test_volume_upload_file_timeout(client, tmp_path, servicer, blob_server, *args):
    call_count = 0

    async def mount_put_file(self, stream):
        await stream.recv_message()
        nonlocal call_count
        call_count += 1
        await stream.send_message(api_pb2.MountPutFileResponse(exists=False))

    with servicer.intercept() as ctx:
        ctx.set_responder("MountPutFile", mount_put_file)
        with mock.patch("modal._utils.blob_utils.LARGE_FILE_LIMIT", 10):
            with mock.patch("modal.volume.VOLUME_PUT_FILE_CLIENT_TIMEOUT", 0.5):
                local_file_path = tmp_path / "bigfile"
                local_file_path.write_text("hello world, this is a lot of text")

                async with modal.Volume.ephemeral(client=client) as vol:
                    with pytest.raises(VolumeUploadTimeoutError):
                        async with vol.batch_upload() as batch:
                            batch.put_file(local_file_path, "/dest")

                assert call_count > 2


@pytest.mark.asyncio
async def test_volume_copy_1(client, tmp_path, servicer):
    ## test 1: copy src path to dst path ##
    src_path = "original.txt"
    dst_path = "copied.txt"
    local_file_path = tmp_path / src_path
    local_file_path.write_text("test copy")

    async with modal.Volume.ephemeral(client=client) as vol:
        # add local file to volume
        async with vol.batch_upload() as batch:
            batch.put_file(local_file_path, src_path)
        object_id = vol.object_id

        # copy file from src_path to dst_path
        vol.copy_files([src_path], dst_path)

    assert servicer.volume_files[object_id].keys() == {src_path, dst_path}

    assert servicer.volume_files[object_id][src_path].data == b"test copy"
    assert servicer.volume_files[object_id][dst_path].data == b"test copy"


@pytest.mark.asyncio
async def test_volume_copy_2(client, tmp_path, servicer):
    ## test 2: copy multiple files into a directory ##
    file_paths = ["file1.txt", "file2.txt"]

    async with modal.Volume.ephemeral(client=client) as vol:
        for file_path in file_paths:
            local_file_path = tmp_path / file_path
            local_file_path.write_text("test copy")
            async with vol.batch_upload() as batch:
                batch.put_file(local_file_path, file_path)
            object_id = vol.object_id

        vol.copy_files(file_paths, "test_dir")

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


@pytest.mark.parametrize("delete_as_instance_method", [True, False])
def test_persisted(servicer, client, delete_as_instance_method):
    # Lookup should fail since it doesn't exist
    with pytest.raises(NotFoundError):
        modal.Volume.lookup("xyz", client=client)

    # Create it
    modal.Volume.lookup("xyz", create_if_missing=True, client=client)

    # Lookup should succeed now
    v = modal.Volume.lookup("xyz", client=client)

    # Delete it
    if delete_as_instance_method:
        with pytest.warns(DeprecationError):
            v.delete()
    else:
        modal.Volume.delete("xyz", client=client)

    # Lookup should fail again
    with pytest.raises(NotFoundError):
        modal.Volume.lookup("xyz", client=client)


def test_ephemeral(servicer, client):
    assert servicer.n_vol_heartbeats == 0
    with modal.Volume.ephemeral(client=client, _heartbeat_sleep=1) as vol:
        assert vol.listdir("/") == []
        # TODO(erikbern): perform some operations
        time.sleep(1.5)  # Make time for 2 heartbeats
    assert servicer.n_vol_heartbeats == 2


def test_lazy_hydration_from_named(set_env_client):
    vol = modal.Volume.from_name("my-vol", create_if_missing=True)
    assert vol.listdir("/") == []


@pytest.mark.skipif(platform.system() != "Linux", reason="needs /proc")
@pytest.mark.asyncio
async def test_open_files_error_annotation(tmp_path):
    assert _open_files_error_annotation(tmp_path) is None

    # Current process keeps file open
    with (tmp_path / "foo.txt").open("w") as _f:
        assert _open_files_error_annotation(tmp_path) == "path foo.txt is open"

    # cwd of current process is inside volume
    cwd = os.getcwd()
    os.chdir(tmp_path)
    assert _open_files_error_annotation(tmp_path) == "cwd is inside volume"
    os.chdir(cwd)

    # Subprocess keeps open file
    open_path = tmp_path / "bar.txt"
    open_path.write_text("")
    proc = await asyncio.create_subprocess_exec("tail", "-f", open_path.as_posix())
    await asyncio.sleep(0.01)  # Give process some time to start
    assert _open_files_error_annotation(tmp_path) == f"path bar.txt is open from 'tail -f {open_path.as_posix()}'"
    proc.kill()
    await proc.wait()
    assert _open_files_error_annotation(tmp_path) is None

    # Subprocess cwd inside volume
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", f"import time; import os; os.chdir('{tmp_path}'); time.sleep(60)"
    )
    # Wait for process to chdir
    for _ in range(100):
        if os.readlink(f"/proc/{proc.pid}/cwd") == tmp_path.as_posix():
            break
        await asyncio.sleep(0.05)
    assert re.match(f"^cwd of '{sys.executable} -c .*' is inside volume$", _open_files_error_annotation(tmp_path))
    proc.kill()
    await proc.wait()
    assert _open_files_error_annotation(tmp_path) is None


@pytest.mark.parametrize("name", ["has space", "has/slash", "a" * 65])
def test_invalid_name(servicer, client, name):
    with pytest.raises(InvalidError, match="Invalid Volume name"):
        modal.Volume.lookup(name)
