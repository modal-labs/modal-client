# Copyright Modal Labs 2024
from modal.file_io import FileIO
from modal_proto import api_pb2

OPEN_EXEC_ID = "exec-open-123"
READ_EXEC_ID = "exec-read-123"
WRITE_EXEC_ID = "exec-write-123"
CLOSE_EXEC_ID = "exec-close-123"


async def container_filesystem_exec(servicer, stream):
    req = await stream.recv_message()

    if req.HasField("file_open_request"):
        # Handle file open
        await stream.send_message(
            api_pb2.ContainerFilesystemExecResponse(exec_id=OPEN_EXEC_ID, file_descriptor="fd-123")
        )
        return

    if req.HasField("file_read_request"):
        # Handle file read
        await stream.send_message(
            api_pb2.ContainerFilesystemExecResponse(
                exec_id=READ_EXEC_ID,
            )
        )
        return

    if req.HasField("file_close_request"):
        # Handle file close
        await stream.send_message(api_pb2.ContainerFilesystemExecResponse(exec_id=CLOSE_EXEC_ID))
        return


def test_file_read(servicer, client):
    """Test file reading."""
    content = "foo\nbar\nbaz\n"
    exec_read_id = "exec-read-123"

    async def container_filesystem_exec_get_output(servicer, stream):
        req = await stream.recv_message()
        if req.exec_id == exec_read_id:
            await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(output=[content.encode()]))
        await stream.send_message(api_pb2.FilesystemRuntimeOutputBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)
        ctx.set_responder("ContainerFilesystemExecGetOutput", container_filesystem_exec_get_output)

        f = FileIO.create("/test.txt", "r", client, "task-123")
        data = f.read()
        f.close()

        assert data == content


def test_file_write(servicer, client):
    """Test file writing."""
    content = "test content\n"

    async def container_filesystem_exec(servicer, stream):
        req = await stream.recv_message()

        if req.HasField("file_open_request"):
            await stream.send_message(
                api_pb2.ContainerFilesystemExecResponse(exec_id="exec-1", file_descriptor="fd-123")
            )
            return

        if req.HasField("file_write_request"):
            assert req.file_write_request.data == content.encode()
            await stream.send_message(api_pb2.ContainerFilesystemExecResponse(exec_id="exec-2"))
            return

        if req.HasField("file_close_request"):
            await stream.send_message(api_pb2.ContainerFilesystemExecResponse(exec_id="exec-3"))
            return

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)

        f = FileIO.create("/test.txt", "w", client, "task-123")
        f.write(content)
        f.close()


def test_file_seek_and_read(servicer, client):
    """Tests that seeking and reading work together correctly."""
    # content = "hello world"

    async def container_filesystem_exec(servicer, stream):
        req = await stream.recv_message()

        if req.HasField("file_open_request"):
            await stream.send_message(
                api_pb2.ContainerFilesystemExecResponse(exec_id="exec-1", file_descriptor="fd-123")
            )
            return

        if req.HasField("file_seek_request"):
            assert req.file_seek_request.offset == 6
            assert req.file_seek_request.whence == api_pb2.SeekWhence.SEEK_SET
            await stream.send_message(api_pb2.ContainerFilesystemExecResponse(exec_id="exec-2"))
            return

        if req.HasField("file_read_request"):
            await stream.send_message(
                api_pb2.ContainerFilesystemExecResponse(
                    exec_id="exec-3",
                )
            )
            return

        if req.HasField("file_close_request"):
            await stream.send_message(api_pb2.ContainerFilesystemExecResponse(exec_id="exec-4"))
            return

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)

        f = FileIO.create("/test.txt", "r", client, "task-123")
        f.seek(6)
        data = f.read()
        f.close()

        assert data == "world"


def test_file_context_manager(servicer, client):
    """Tests that the file context manager works correctly."""
    content = "test content\n"

    async def container_filesystem_exec(servicer, stream):
        req = await stream.recv_message()

        if req.HasField("file_open_request"):
            await stream.send_message(
                api_pb2.ContainerFilesystemExecResponse(exec_id="exec-1", file_descriptor="fd-123")
            )
            return

        if req.HasField("file_read_request"):
            await stream.send_message(
                api_pb2.ContainerFilesystemExecResponse(
                    exec_id="exec-2",
                )
            )
            return

        if req.HasField("file_close_request"):
            await stream.send_message(api_pb2.ContainerFilesystemExecResponse(exec_id="exec-3"))
            return

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerFilesystemExec", container_filesystem_exec)

        with FileIO.create("/test.txt", "r", client, "task-123") as f:
            data = f.read()

        assert data == content
