# Copyright Modal Labs 2024
from modal import enable_output
from modal.io_streams import StreamReader
from modal_proto import api_pb2


def test_stream_reader(servicer, client):
    lines = ["foo\n", "bar\n", "baz\n"]

    async def sandbox_get_logs(servicer, stream):
        await stream.recv_message()

        # Data with multiple lines
        log1 = api_pb2.TaskLogs(
            data="".join(lines),
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
        )
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="0", items=[log1]))

        # Data with single lines sent separately
        for i, line in enumerate(lines):
            log = api_pb2.TaskLogs(data=line, file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
            await stream.send_message(api_pb2.TaskLogsBatch(entry_id=str(i + 1), items=[log]))

        # Send EOF
        await stream.send_message(api_pb2.TaskLogsBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxGetLogs", sandbox_get_logs)

        with enable_output():
            stdout = StreamReader(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                object_id="sb-123",
                object_type="sandbox",
                client=client,
            )

            out = []
            for line in stdout:
                out.append(line)

            assert out == lines * 2
