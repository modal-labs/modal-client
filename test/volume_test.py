# Copyright Modal Labs 2023
import asyncio
import io
import os
import platform
import pytest
import random
import re
import sys
import time
from difflib import ndiff
from pathlib import Path
from unittest import mock

import modal
from modal._utils.blob_utils import BLOCK_SIZE
from modal.exception import AlreadyExistsError, DeprecationError, InvalidError, NotFoundError, VolumeUploadTimeoutError
from modal.volume import _open_files_error_annotation
from modal_proto import api_pb2

VERSIONS = [
    None,
    api_pb2.VOLUME_FS_VERSION_V2,
]


def dummy():
    pass


def assert_eq_large(left, right):
    assert len(left) == len(right)
    if left != right:
        raise AssertionError(ndiff(left.splitlines(), right.splitlines()))


def test_volume_info(servicer, client):
    name = "super-important-data"
    vol = modal.Volume.from_name(name, create_if_missing=True)
    assert vol.name == name

    vol.hydrate(client)
    info = vol.info()
    assert info.name == name
    assert info.created_by == servicer.default_username


@pytest.mark.parametrize("read_only", [True, False])
@pytest.mark.parametrize("version", VERSIONS)
def test_volume_mount(client, servicer, version, read_only):
    app = modal.App()

    vol = modal.Volume.from_name("xyz", create_if_missing=True, version=version)
    if read_only:
        vol = vol.read_only()

    _ = app.function(volumes={"/root/foo": vol})(dummy)

    with servicer.intercept() as ctx:
        with app.run(client=client):
            req = ctx.pop_request("FunctionCreate")
            assert req.function.volume_mounts[0].read_only == read_only


def test_volume_bad_paths():
    app = modal.App()
    vol = modal.Volume.from_name("xyz")

    with pytest.raises(InvalidError):
        app.function(volumes={"/root/../../foo": vol})(dummy)

    with pytest.raises(InvalidError):
        app.function(volumes={"/": vol})(dummy)

    with pytest.raises(InvalidError):
        app.function(volumes={"/tmp/": vol})(dummy)


def test_volume_mount_read_only_error(client):
    read_only_vol = modal.Volume.from_name("xyz", create_if_missing=True).read_only()
    read_only_vol_hydrated = read_only_vol.hydrate(client)

    with pytest.raises(InvalidError):
        read_only_vol_hydrated.batch_upload()

    with pytest.raises(InvalidError):
        read_only_vol_hydrated.remove_file("file1.txt")

    with pytest.raises(InvalidError):
        read_only_vol_hydrated.copy_files(["file1.txt"], "bar2")


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
@pytest.mark.skip(reason="TODO(dflemstr) this test has started flaking at a high rate recently")
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("file_contents_size", [100, 8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024 + 4711])
async def test_volume_get(servicer, client, tmp_path, version, file_contents_size):
    await modal.Volume.objects.create.aio("my-vol", client=client, version=version)
    vol = await modal.Volume.from_name("my-vol").hydrate.aio(client=client)

    file_contents = random.randbytes(file_contents_size)
    file_path = "foo.bin"
    local_file_path = tmp_path / file_path
    local_file_path.write_bytes(file_contents)

    async with vol.batch_upload() as batch:
        batch.put_file(local_file_path, file_path)

    data = b""
    for chunk in vol.read_file(file_path):
        data += chunk

    # Faster assert to avoid huge error when there are large content differences:
    assert len(data) == file_contents_size
    assert data == file_contents

    output = io.BytesIO()
    vol.read_file_into_fileobj(file_path, output)

    # Faster assert to avoid huge error when there are large content differences:
    assert len(output.getvalue()) == file_contents_size
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
@pytest.mark.parametrize("version", VERSIONS)
async def test_volume_batch_upload(servicer, client, tmp_path, version):
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    local_dir = tmp_path / "some_dir"
    local_dir.mkdir()
    (local_dir / "smol").write_text("###")

    subdir = local_dir / "subdir"
    subdir.mkdir()
    (subdir / "other").write_text("####")

    async with modal.Volume.ephemeral(client=client, version=version) as vol:
        with open(local_file_path, "rb") as fp:
            with vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/some_file")
                batch.put_directory(local_dir, "/some_dir")
                batch.put_file(io.BytesIO(b"data from a file-like object"), "/filelike", mode=0o600)
                batch.put_directory(local_dir, "/non-recursive", recursive=False)
                batch.put_file(fp, "/filelike2")
        object_id = vol.object_id

    assert servicer.volumes[object_id].files.keys() == {
        "/some_file",
        "/some_dir/smol",
        "/some_dir/subdir/other",
        "/filelike",
        "/non-recursive/smol",
        "/filelike2",
    }
    assert servicer.volumes[object_id].files["/some_file"].data == b"hello world"
    assert servicer.volumes[object_id].files["/some_dir/smol"].data == b"###"
    assert servicer.volumes[object_id].files["/some_dir/subdir/other"].data == b"####"
    assert servicer.volumes[object_id].files["/filelike"].data == b"data from a file-like object"
    assert servicer.volumes[object_id].files["/filelike"].mode == 0o600
    assert servicer.volumes[object_id].files["/non-recursive/smol"].data == b"###"
    assert servicer.volumes[object_id].files["/filelike2"].data == b"hello world"
    assert servicer.volumes[object_id].files["/filelike2"].mode == 0o644


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_volume_batch_upload_bytesio(servicer, client, tmp_path, version):
    async with modal.Volume.ephemeral(client=client, version=version) as vol:
        with vol.batch_upload() as batch:
            batch.put_file(io.BytesIO(b"data from a file-like object"), "/filelike", mode=0o600)
        object_id = vol.object_id

    assert servicer.volumes[object_id].files.keys() == {
        "/filelike",
    }
    assert servicer.volumes[object_id].files["/filelike"].data == b"data from a file-like object"
    assert servicer.volumes[object_id].files["/filelike"].mode == 0o600


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_volume_batch_upload_opened_file(servicer, client, tmp_path, version):
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    async with modal.Volume.ephemeral(client=client, version=version) as vol:
        with open(local_file_path, "rb") as fp, vol.batch_upload() as batch:
            batch.put_file(fp, "/filelike2", mode=0o600)
        object_id = vol.object_id

    assert servicer.volumes[object_id].files.keys() == {
        "/filelike2",
    }
    assert servicer.volumes[object_id].files["/filelike2"].data == b"hello world"
    assert servicer.volumes[object_id].files["/filelike2"].mode == 0o600


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_volume_batch_upload_force(servicer, client, tmp_path, version):
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    local_file_path2 = tmp_path / "some_file2"
    local_file_path2.write_text("overwritten")

    async with modal.Volume.ephemeral(client=client, version=version) as vol:
        with servicer.intercept() as ctx:
            # Seed the volume
            with vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/some_file")

            if version == api_pb2.VOLUME_FS_VERSION_V2:
                # The batch should involve two calls; once with a missing_blocks response
                assert ctx.pop_request("VolumePutFiles2").disallow_overwrite_existing_files
                assert ctx.pop_request("VolumePutFiles2").disallow_overwrite_existing_files
            else:
                assert ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

            # Attempting to overwrite the file with force=False should result in an error
            with pytest.raises(FileExistsError):
                with vol.batch_upload(force=False) as batch:
                    batch.put_file(local_file_path, "/some_file")

            if version == api_pb2.VOLUME_FS_VERSION_V2:
                # The batch should fail on the first call since force=False
                assert ctx.pop_request("VolumePutFiles2").disallow_overwrite_existing_files
            else:
                assert ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

            assert servicer.volumes[vol.object_id].files["/some_file"].data == b"hello world"

            # Overwriting should work with force=True
            with vol.batch_upload(force=True) as batch:
                batch.put_file(local_file_path2, "/some_file")

            if version == api_pb2.VOLUME_FS_VERSION_V2:
                # The batch should involve two calls; once with a missing_blocks response
                assert not ctx.pop_request("VolumePutFiles2").disallow_overwrite_existing_files
                assert not ctx.pop_request("VolumePutFiles2").disallow_overwrite_existing_files
            else:
                assert not ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

            assert servicer.volumes[vol.object_id].files["/some_file"].data == b"overwritten"


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_volume_upload_removed_file(servicer, client, tmp_path, version):
    local_file_path = tmp_path / "some_file"
    local_file_path.write_text("hello world")

    async with modal.Volume.ephemeral(client=client, version=version) as vol:
        with pytest.raises(FileNotFoundError):
            with vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/dest")
                local_file_path.unlink()


@pytest.mark.asyncio
async def test_volume_upload_large_file(client, tmp_path, servicer, blob_server):
    with mock.patch("modal._utils.blob_utils.LARGE_FILE_LIMIT", 10):
        local_file_path = tmp_path / "bigfile"
        local_file_path.write_text("hello world, this is a lot of text")

        async with modal.Volume.ephemeral(client=client) as vol:
            async with vol.batch_upload() as batch:
                batch.put_file(local_file_path, "/a")
            object_id = vol.object_id

        assert servicer.volumes[object_id].files.keys() == {"/a"}
        assert servicer.volumes[object_id].files["/a"].data == b""
        assert servicer.volumes[object_id].files["/a"].data_blob_id == "bl-1"

        _, blobs, _, _ = blob_server
        assert blobs["bl-1"] == b"hello world, this is a lot of text"


@pytest.mark.asyncio
async def test_volume2_upload_large_file(client, tmp_path, servicer, blob_server):
    # Volumes version 2 don't use `modal._utils.blob_utils.LARGE_FILE_LIMIT`
    # Instead, we need to go over 8MiB to trigger different behavior (ie spilling into multiple blocks), but in this
    # unit test context there isn't much of a semantic difference.

    # Create a byte buffer that is larger than 8MiB
    data = b"hello world, this is a lot of text" * 250_000
    assert len(data) > BLOCK_SIZE

    local_file_path = tmp_path / "bigfile"
    local_file_path.write_bytes(data)

    async with modal.Volume.ephemeral(client=client, version=api_pb2.VOLUME_FS_VERSION_V2) as vol:
        async with vol.batch_upload() as batch:
            batch.put_file(local_file_path, "/a")
        object_id = vol.object_id

    assert servicer.volumes[object_id].files.keys() == {"/a"}
    # Volumes version 2 don't use blob entities
    assert_eq_large(servicer.volumes[object_id].files["/a"].data, data)


@pytest.mark.asyncio
async def test_volume2_upload_large_blank_file(client, tmp_path, servicer, blob_server):
    # Volumes version 2 don't use `modal._utils.blob_utils.LARGE_FILE_LIMIT`
    # Instead, we need to go over 8MiB to trigger different behavior (ie spilling into multiple blocks), but in this
    # unit test context there isn't much of a semantic difference.

    # Create a byte buffer that is larger than 8MiB. Each block starts with b"a" followed by zeroes until the next
    # block boundary, except the last block that just contains b"cdef"
    data = (b"a" + (b"\0" * (8 * 1024 * 1024 - 1))) * 2 + b"cdef"
    assert len(data) > BLOCK_SIZE

    local_file_path = tmp_path / "bigfile"
    local_file_path.write_bytes(data)

    async with modal.Volume.ephemeral(client=client, version=api_pb2.VOLUME_FS_VERSION_V2) as vol:
        async with vol.batch_upload() as batch:
            batch.put_file(local_file_path, "/a")
        object_id = vol.object_id

    assert servicer.volumes[object_id].files.keys() == {"/a"}
    # Volumes version 2 don't use blob entities
    assert_eq_large(servicer.volumes[object_id].files["/a"].data, data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        b"abc",
        b"\0\0\0",  # <- this is the only case that works fine, we maybe don't do anything?
        b"\0\0\0abc",
        b"abc\0\0\0def",
        b"abc\0\0\0",
    ],
)
async def test_volume2_upload_small_blank_file(client, tmp_path, servicer, blob_server, data):
    local_file_path = tmp_path / "smallfile"
    local_file_path.write_bytes(data)

    async with modal.Volume.ephemeral(client=client, version=api_pb2.VOLUME_FS_VERSION_V2) as vol:
        async with vol.batch_upload() as batch:
            batch.put_file(local_file_path, "/a")
        object_id = vol.object_id

    assert servicer.volumes[object_id].files.keys() == {"/a"}
    assert_eq_large(servicer.volumes[object_id].files["/a"].data, data)


@pytest.mark.asyncio
async def test_volume_upload_large_stream(client, servicer, blob_server):
    with mock.patch("modal._utils.blob_utils.LARGE_FILE_LIMIT", 10):
        stream = io.BytesIO(b"hello world, this is a lot of text")

        async with modal.Volume.ephemeral(client=client) as vol:
            async with vol.batch_upload() as batch:
                batch.put_file(stream, "/a")
            object_id = vol.object_id

        assert servicer.volumes[object_id].files.keys() == {"/a"}
        assert servicer.volumes[object_id].files["/a"].data == b""
        assert servicer.volumes[object_id].files["/a"].data_blob_id == "bl-1"

        _, blobs, _, _ = blob_server
        assert blobs["bl-1"] == b"hello world, this is a lot of text"


@pytest.mark.asyncio
async def test_volume2_upload_large_stream(client, servicer, blob_server):
    # Volumes version 2 don't use `modal._utils.blob_utils.LARGE_FILE_LIMIT`
    # Instead, we need to go over 8MiB to trigger different behavior (ie spilling into multiple blocks), but in this
    # unit test context there isn't much of a semantic difference.

    # Create a byte buffer that is larger than 8MiB
    data = b"hello world, this is a lot of text" * 250_000
    assert len(data) > BLOCK_SIZE

    stream = io.BytesIO(data)

    async with modal.Volume.ephemeral(client=client, version=api_pb2.VOLUME_FS_VERSION_V2) as vol:
        async with vol.batch_upload() as batch:
            batch.put_file(stream, "/a")
        object_id = vol.object_id

    assert servicer.volumes[object_id].files.keys() == {"/a"}
    assert servicer.volumes[object_id].files["/a"].data == data


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
@pytest.mark.parametrize("version", VERSIONS)
async def test_volume_copy_1(client, tmp_path, servicer, version):
    ## test 1: copy src path to dst path ##
    src_path = "original.txt"
    dst_path = "copied.txt"
    local_file_path = tmp_path / src_path
    local_file_path.write_text("test copy")

    async with modal.Volume.ephemeral(client=client, version=version) as vol:
        # add local file to volume
        async with vol.batch_upload() as batch:
            batch.put_file(local_file_path, src_path)
        object_id = vol.object_id

        # copy file from src_path to dst_path
        vol.copy_files([src_path], dst_path, False)

    assert servicer.volumes[object_id].files.keys() == {src_path, dst_path}

    assert servicer.volumes[object_id].files[src_path].data == b"test copy"
    assert servicer.volumes[object_id].files[dst_path].data == b"test copy"


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_volume_copy_2(client, tmp_path, servicer, version):
    ## test 2: copy multiple files into a directory ##
    file_paths = ["file1.txt", "file2.txt"]

    async with modal.Volume.ephemeral(client=client, version=version) as vol:
        for file_path in file_paths:
            local_file_path = tmp_path / file_path
            local_file_path.write_text("test copy")
            async with vol.batch_upload() as batch:
                batch.put_file(local_file_path, file_path)
            object_id = vol.object_id

        vol.copy_files(file_paths, "test_dir", False)

    returned_volume_files = [Path(file) for file in servicer.volumes[object_id].files.keys()]
    expected_volume_files = [
        Path(file) for file in ["file1.txt", "file2.txt", "test_dir/file1.txt", "test_dir/file2.txt"]
    ]

    assert returned_volume_files == expected_volume_files

    returned_file_data = {
        Path(entry): servicer.volumes[object_id].files[entry] for entry in servicer.volumes[object_id].files
    }
    assert returned_file_data[Path("test_dir/file1.txt")].data == b"test copy"
    assert returned_file_data[Path("test_dir/file2.txt")].data == b"test copy"


@pytest.mark.parametrize("version", VERSIONS)
def test_from_name(servicer, client, version):
    # Lookup should fail since it doesn't exist
    with pytest.raises(NotFoundError):
        modal.Volume.from_name("xyz", version=version).hydrate(client)

    # Create it
    modal.Volume.from_name("xyz", create_if_missing=True, version=version).hydrate(client)

    # Lookup should succeed now
    modal.Volume.from_name("xyz", version=version).hydrate(client)

    modal.Volume.objects.delete("xyz", client=client)
    # Lookup should fail again
    with pytest.raises(NotFoundError):
        modal.Volume.from_name("xyz", version=version).hydrate(client)
    modal.Volume.objects.delete("xyz", client=client, allow_missing=True)


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
def test_invalid_name(name):
    with pytest.raises(InvalidError, match="Invalid Volume name"):
        modal.Volume.from_name(name)


@pytest.fixture()
def unset_main_thread_event_loop():
    try:
        event_loop = asyncio.get_event_loop()
    except RuntimeError:
        event_loop = None

    asyncio.set_event_loop(None)
    try:
        yield
    finally:
        asyncio.set_event_loop(event_loop)  # reset so we don't break other tests


@pytest.mark.usefixtures("unset_main_thread_event_loop")
def test_lock_is_py39_safe(set_env_client):
    vol = modal.Volume.from_name("my_vol", create_if_missing=True)
    vol.reload()


def test_volume_namespace_deprecated(servicer, client):
    # Test from_name with namespace parameter warns
    with pytest.warns(
        DeprecationError,
        match="The `namespace` parameter for `modal.Volume.from_name` is deprecated",
    ):
        modal.Volume.from_name("test-volume", namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE)

    # Test that from_name without namespace parameter doesn't warn about namespace
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        modal.Volume.from_name("test-volume")
    # Filter out any unrelated warnings
    namespace_warnings = [w for w in record if "namespace" in str(w.message).lower()]
    assert len(namespace_warnings) == 0


def test_remove_file_not_found(set_env_client):
    vol = modal.Volume.from_name("my_vol", create_if_missing=True)
    with pytest.raises(FileNotFoundError):
        vol.remove_file("a")


def test_volume_list(servicer, client):
    for i in range(5):
        modal.Volume.from_name(f"test-volume-{i}", create_if_missing=True).hydrate(client)
    if sys.platform == "win32":
        time.sleep(1 / 32)

    volume_list = modal.Volume.objects.list(client=client)
    assert len(volume_list) == 5
    assert all(v.name.startswith("test-volume-") for v in volume_list)
    assert all(v.info().created_by == servicer.default_username for v in volume_list)

    volume_list = modal.Volume.objects.list(max_objects=2, client=client)
    assert len(volume_list) == 2


def test_volume_create(servicer, client):
    modal.Volume.objects.create(name="test-volume-create", client=client)
    modal.Volume.from_name("test-volume-create").hydrate(client)
    with pytest.raises(AlreadyExistsError):
        modal.Volume.objects.create(name="test-volume-create", client=client)
    modal.Volume.objects.create(name="test-volume-create", allow_existing=True, client=client)
    with pytest.raises(InvalidError, match="Invalid Volume name"):
        modal.Volume.objects.create(name="has space", client=client)


def test_volume_create_version(servicer, client):
    for version in [1, 2]:
        modal.Volume.objects.create(name=f"should-be-v{version}", version=version, client=client)
        vol_id = servicer.deployed_volumes[(f"should-be-v{version}", "main")]
        assert servicer.volumes[vol_id].version == version

    with pytest.raises(InvalidError, match="VolumeFS version must be either 1 or 2"):
        modal.Volume.objects.create(name="should-be-v3", version=3, client=client)
