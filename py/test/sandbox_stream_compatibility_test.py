# Copyright Modal Labs 2025
"""Compatibility checks for streams that use the shared server connection."""

import pytest

from modal.sandbox import Sandbox
from modal_proto import api_pb2

SANDBOX_ID = "sb-nGEijt9WbBMlGrsPH9FOaC"


@pytest.mark.asyncio
@pytest.mark.parametrize("descriptor", ["stdout", "stderr"])
async def test_held_server_output_reader_can_reopen_after_detach(servicer, client, descriptor):
    requests = []

    async def logs(servicer, stream):
        request = await stream.recv_message()
        requests.append(request)
        await stream.send_message(
            api_pb2.TaskLogsBatch(
                entry_id=str(len(requests)),
                items=[api_pb2.TaskLogs(data="output\n", file_descriptor=request.file_descriptor)],
                eof=len(requests) == 2,
            )
        )

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxGetLogs", logs)
        sandbox = await Sandbox.from_id.aio(SANDBOX_ID, client=client)
        reader = getattr(sandbox, descriptor)
        await sandbox.detach.aio()

        assert await reader.read.aio() == "output\noutput\n"
        assert [request.last_entry_id for request in requests] == ["", "1"]


@pytest.mark.asyncio
async def test_held_server_stdin_writer_remains_usable_after_detach(servicer, client):
    sandbox = await Sandbox.from_id.aio(SANDBOX_ID, client=client)
    writer = sandbox.stdin
    writer.write(b"before")
    await sandbox.detach.aio()

    writer.write(b"after")
    writer.write_eof()
    with servicer.intercept() as ctx:
        await writer.drain.aio()
        (request,) = ctx.get_requests("SandboxStdinWrite")
        assert request.input == b"beforeafter"
        assert request.eof
