# Copyright Modal Labs 2024
import pytest

from grpclib import Status
from grpclib.exceptions import GRPCError

from modal.file_io import WRITE_CHUNK_SIZE, WRITE_FILE_SIZE_LIMIT, FileIO, delete_bytes, replace_bytes  # type: ignore
from modal_proto import api_pb2

OPEN_EXEC_ID = "exec-open-123"
READ_EXEC_ID = "exec-read-123"
READLINE_EXEC_ID = "exec-readline-123"
READLINES_EXEC_ID = "exec-readlines-123"
WRITE_EXEC_ID = "exec-write-123"
FLUSH_EXEC_ID = "exec-flush-123"
SEEK_EXEC_ID = "exec-seek-123"
WRITE_REPLACE_EXEC_ID = "exec-write-replace-123"
DELETE_EXEC_ID = "exec-delete-123"
CLOSE_EXEC_ID = "exec-close-123"
LS_EXEC_ID = "exec-ls-123"
MKDIR_EXEC_ID = "exec-mkdir-123"
RM_EXEC_ID = "exec-rm-123"


async def container_filesystem_exec(servicer, stream):
    req = await stream.recv_message()

    if req.HasField("file_open_request"):
        await stream.send_message(
            api_pb2.ContainerFilesystemExecResponse(exec_id=OPEN_EXEC_ID, file_descriptor="fd-123")
        )
    elif req.HasField("file_read_request"):
        await stream.send_message(
            api_pb2.ContainerFilesystemExecResponse(
                exec_id=READ_EXEC_ID,
            )
        )
    elif req.HasField("file_read_line_request"):
        await stream.send_message(
            api_pb2.ContainerFilesystemExecResponse(
                exec_id=READLINE_EXEC_ID,
            )
        )
    elif req.HasField("file_write_request"):
        await stream.send_message(
            api_pb2.ContainerFilesystemExecResponse(
                exec_id=WRITE_EXEC_ID,
            )
        )
    elif req.HasField("file_write_replace_bytes_request"):
        await stream.send_message(
            api_pb2.ContainerFilesystemExecResponse(
                exec_id=WRITE_REPLACE_EXEC_ID,
            )
        )
    elif req.HasField("file_delete_bytes_request"):
        await stream.send_message(
            api_pb2.ContainerFilesystemExecResponse(
                exec_id=DELETE_EXEC_ID,
            )
        )
    elif req.HasField("file_seek_request"):
        await stream.send_message(
            api_pb2.ContainerFilesystemExecResponse(
                exec_id=SEEK_EXEC_ID,
            )
        )
    elif req.HasField("file_flush_request"):
        await stream.send_message(
            api_pb2.ContainerFilesystemExecResponse(
                exec_id=FLUSH_EXEC_ID,
            )
        )
    elif req.HasField("file_close_request"):
        await stream.send_message(api_pb2.ContainerFilesystemExecResponse(exec_id=CLOSE_EXEC_ID))
    elif req.HasField("file_ls_request"):
        await stream.send_message(api_pb2.ContainerFilesystemExecResponse(exec_id=LS_EXEC_ID))
    elif req.HasField("file_mkdir_request"):
        await stream.send_message(api_pb2.ContainerFilesystemExecResponse(exec_id=MKDIR_EXEC_ID))
    elif req.HasField("file_rm_request"):
        await stream.send_message(api_pb2.ContainerFilesystemExecResponse(exec_id=RM_EXEC_ID))


def test_file_read(servicer, client):
    """Test file reading."""
    content = "foo\nbar\nbaz\n"

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[content.encode()]))
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "r", client, "task-123")
        assert f.read() == content
        f.close()


def test_file_write(servicer, client):
    """Test file writing."""
    content = "foo\nbar\nbaz\n"

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[content.encode()]))
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "a+", client, "task-123")
        f.write(content)
        assert f.read() == content
        f.close()


def test_file_write_large(servicer, client):
    """Test file write chunking logic."""
    content = "A" * WRITE_FILE_SIZE_LIMIT
    write_counter = 0

    async def container_filesystem_exec_get_output(servicer, stream):
        nonlocal write_counter
        req = await stream.recv_message()
        if req.exec_id == WRITE_EXEC_ID:
            write_counter += 1
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "a+", client, "task-123")
        f.write(content)
        assert write_counter == WRITE_FILE_SIZE_LIMIT // WRITE_CHUNK_SIZE
        f.close()


def test_file_write_too_large(servicer, client):
    """Test that writing a file larger than WRITE_FILE_SIZE_LIMIT raises an error."""
    content = "A" * (WRITE_FILE_SIZE_LIMIT + 1)

    async def container_filesystem_exec_get_output(servicer, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        with pytest.raises(ValueError):
            FileIO.create("/test.txt", "a+", client, "task-123").write(content)


def test_file_readline(servicer, client):
    """Test file reading line by line."""
    lines = ["foo\n", "bar\n", "baz\n", "end"]
    content = "".join(lines)
    index = 0

    async def container_filesystem_exec_get_output(servicer, stream):
        nonlocal index
        req = await stream.recv_message()
        if req.exec_id == READLINE_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[lines[index].encode()]))
            index += 1
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "w+", client, "task-123")
        f.write(content)
        assert f.readline() == "foo\n"
        assert f.readline() == "bar\n"
        assert f.readline() == "baz\n"
        assert f.readline() == "end"
        f.close()


def test_file_readlines_no_newline_end(servicer, client):
    """Test file reading lines."""
    lines = ["foo\n", "bar\n", "baz\n", "end"]
    content = "".join(lines)

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[content.encode()]))
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "w+", client, "task-123")
        f.write(content)
        assert f.readlines() == lines
        f.close()


def test_file_readlines_newline_end(servicer, client):
    """Test file reading lines."""
    lines = ["foo\n", "bar\n", "baz\n", "end\n"]
    content = "".join(lines)

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[content.encode()]))
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "w+", client, "task-123")
        f.write(content)
        assert f.readlines() == lines
        f.close()


def test_file_readlines_multiple_newline_end(servicer, client):
    """Test file reading lines."""
    lines = ["foo\n", "bar\n", "baz\n", "end\n", "\n", "\n"]
    content = "".join(lines)

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[content.encode()]))
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "w+", client, "task-123")
        f.write(content)
        assert f.readlines() == lines
        f.close()


def test_file_flush(servicer, client):
    """Test file flushing."""

    async def container_filesystem_exec_get_output(servicer, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "w+", client, "task-123")
        f.write("foo")
        f.flush()
        f.close()


def test_file_seek(servicer, client):
    """Test file seeking."""
    index = 0
    expected_outputs = ["foo\nbar\nbaz\n", "bar\nbaz\n", "baz\n", ""]

    async def container_filesystem_exec_get_output(servicer, stream):
        nonlocal index
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[expected_outputs[index].encode()]))
            index += 1
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "a+", client, "task-123")
        f.write("foo\nbar\nbaz\n")
        f.close()
        f = FileIO.create("/test.txt", "r", client, "task-123")
        for i in range(4):
            f.seek(3)
            assert f.read() == expected_outputs[i]
        f.close()


def test_file_write_replace_bytes(servicer, client):
    """Test file write replace bytes."""
    index = 0
    expected_outputs = ["foo\nbar\nbaz\n", "foo\nbarbar\nbaz\n"]

    async def container_filesystem_exec_get_output(servicer, stream):
        nonlocal index
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[expected_outputs[index].encode()]))
            index += 1
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "a+", client, "task-123")
        f.write("foo\nbar\nbaz\n")
        assert f.read() == expected_outputs[0]
        replace_bytes(f, data=b"barbar", start=4, end=7)
        assert f.read() == expected_outputs[1]
        f.close()


def test_file_delete_bytes(servicer, client):
    """Test file delete bytes."""
    index = 0
    expected_outputs = ["foo\nbar\nbaz\n", "foo\nbaz\n"]

    async def container_filesystem_exec_get_output(servicer, stream):
        nonlocal index
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[expected_outputs[index].encode()]))
            index += 1
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "a+", client, "task-123")
        f.write("foo\nbar\nbaz\n")
        assert f.read() == expected_outputs[0]
        delete_bytes(f, start=4, end=7)
        assert f.read() == expected_outputs[1]
        f.close()


def test_invalid_mode(servicer, client):
    """Test a variety of invalid modes."""
    invalid_modes = [
        "",  # empty mode
        "invalid",  # invalid mode
        "rr",  # duplicate letters
        "rab",  # too many modes
        "r++",  # too many modes
        "+",  # plus without read/write mode
        "x+r",  # too many modes
        "wx",  # too many modes
        "rbb",  # too many binary flags
        " r",  # whitespace
        "r ",  # whitespace
        "R",  # uppercase
        "W",  # uppercase
        "\n",  # newline
    ]
    for mode in invalid_modes:
        with pytest.raises(ValueError):
            FileIO.create("/test.txt", mode, client, "task-123")  # type: ignore


def test_client_retry(servicer, client):
    """Test client retry."""
    retries = 5
    content = "foo\nbar\nbaz\n"

    async def container_filesystem_exec_get_output(servicer, stream):
        nonlocal retries
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            if retries > 0:
                retries -= 1
                raise GRPCError(Status.UNAVAILABLE, "test")
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[content.encode()]))
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "w+", client, "task-123")
        f.write(content)
        assert f.read() == content
        f.close()


@pytest.mark.asyncio
async def test_file_io_async_context_manager(servicer, client):
    """Test file io context manager."""
    content = "foo\nbar\nbaz\n"

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[content.encode()]))
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        async with FileIO.create("/test.txt", "w+", client, "task-123") as f:
            await f.write.aio(content)
            assert await f.read.aio() == content


def test_file_io_sync_context_manager(servicer, client):
    """Test file io context manager."""
    content = "foo\nbar\nbaz\n"

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == READ_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[content.encode()]))
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        with FileIO.create("/test.txt", "w+", client, "task-123") as f:
            f.write(content)
            assert f.read() == content


def test_ls(servicer, client):
    """Test ls."""

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == LS_EXEC_ID:
            await stream.send_message(
                api_pb2.FilesystemRuntimeOutputBatch(output=[b'{"paths": ["foo", "bar", "baz"]}'])
            )
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))
        else:
            raise Exception("Unexpected exec_id: " + req.exec_id)

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        files = FileIO.ls("/test.txt", client, "task-123")
        assert files == ["foo", "bar", "baz"]


def test_mkdir(servicer, client):
    """Test mkdir."""

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == MKDIR_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))
        else:
            raise Exception("Unexpected exec_id: " + req.exec_id)

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        FileIO.mkdir("/test.txt", client, "task-123")


def test_rm(servicer, client):
    """Test rm."""

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == RM_EXEC_ID:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))
        else:
            raise Exception("Unexpected exec_id: " + req.exec_id)

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        FileIO.rm("/test.txt", client, "task-123")
