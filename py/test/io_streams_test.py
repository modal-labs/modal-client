# Copyright Modal Labs 2024
import pytest
import time
import typing
from typing import AsyncGenerator, Optional

from modal import enable_output
from modal._utils.async_utils import aclosing, sync_or_async_iter, synchronizer
from modal._utils.task_command_router_client import TaskCommandRouterClient
from modal.io_streams import (
    StreamReader,
    _decode_bytes_stream_to_str,
    _stream_by_line,
    _StreamReaderThroughSandboxCommandRouterParams,
    _StreamReaderThroughSandboxExecCommandRouterParams,
    _StreamReaderThroughServerParams,
    _StreamWriter,
    _StreamWriterThroughCommandRouterParams,
)
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2


def _build_stream_reader_params(
    *,
    file_descriptor,
    object_id,
    client=None,
    command_router_client=None,
    task_id=None,
    deadline=None,
):
    """Helper: build the right params dataclass for a stream reader test."""
    if command_router_client is not None:
        return _StreamReaderThroughSandboxExecCommandRouterParams(
            file_descriptor=file_descriptor,
            task_id=task_id,
            object_id=object_id,
            command_router_client=command_router_client,
            deadline=deadline,
        )
    return _StreamReaderThroughServerParams(
        file_descriptor=file_descriptor,
        object_id=object_id,
        client=client,
    )


@synchronizer.wrap
async def _make_stream_reader(**kwargs) -> StreamReader:
    # helper to make sure stream readers are constructed in the synchronizer event loop
    params_kwargs = {
        k: kwargs.pop(k, None)
        for k in (
            "file_descriptor",
            "object_id",
            "client",
            "command_router_client",
            "task_id",
            "deadline",
        )
    }
    params = _build_stream_reader_params(**params_kwargs)
    return StreamReader(params, **kwargs)


def make_stream_reader(**kwargs) -> StreamReader:
    # stupid wrapper for type safety, otherwise the interpreter thinks the wrapper is async above
    # and we need to type ignore everywhere
    return typing.cast(StreamReader, _make_stream_reader(**kwargs))


def test_stream_reader(servicer, client):
    """Tests that the stream reader works with clean inputs."""
    lines = ["foo\n", "bar\n", "baz\n"]

    async def sandbox_get_logs(servicer, stream):
        await stream.recv_message()

        for line in lines:
            log = api_pb2.TaskLogs(data=line, file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
            await stream.send_message(api_pb2.TaskLogsBatch(entry_id=line, items=[log]))

        # send EOF
        await stream.send_message(api_pb2.TaskLogsBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxGetLogs", sandbox_get_logs)

        with enable_output():
            stdout: StreamReader[str] = make_stream_reader(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                object_id="sb-123",
                client=client,
            )

            out = []
            for line in stdout:
                out.append(line)

            assert out == lines


def test_stream_reader_processed(servicer, client):
    """Tests that the stream reader with logs by line works with clean inputs."""
    lines = ["foo\n", "bar\n", "baz\n"]

    async def sandbox_get_logs(servicer, stream):
        await stream.recv_message()

        for line in lines:
            log = api_pb2.TaskLogs(data=line, file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
            await stream.send_message(api_pb2.TaskLogsBatch(entry_id=line, items=[log]))

        # send EOF
        await stream.send_message(api_pb2.TaskLogsBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxGetLogs", sandbox_get_logs)

        with enable_output():
            stdout: StreamReader[str] = make_stream_reader(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                object_id="sb-123",
                client=client,
                by_line=True,
            )

            out = []
            for line in stdout:
                out.append(line)

            assert out == lines


def test_stream_reader_processed_multiple(servicer, client):
    """Tests that the stream reader with logs by line splits multiple lines."""

    async def sandbox_get_logs(servicer, stream):
        await stream.recv_message()

        log = api_pb2.TaskLogs(
            data="foo\nbar\nbaz",
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
        )
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="0", items=[log]))

        # send EOF
        await stream.send_message(api_pb2.TaskLogsBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxGetLogs", sandbox_get_logs)

        with enable_output():
            stdout: StreamReader[str] = make_stream_reader(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                object_id="sb-123",
                client=client,
                by_line=True,
            )

            out = []
            for line in stdout:
                out.append(line)

            assert out == ["foo\n", "bar\n", "baz"]


def test_stream_reader_processed_partial_lines(servicer, client):
    """Test that the stream reader with logs by line joins partial lines together."""

    async def sandbox_get_logs(servicer, stream):
        await stream.recv_message()

        log1 = api_pb2.TaskLogs(
            data="foo",
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
        )
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="0", items=[log1]))

        log2 = api_pb2.TaskLogs(
            data="bar\n",
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
        )
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="1", items=[log2]))

        log3 = api_pb2.TaskLogs(
            data="baz",
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
        )
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="2", items=[log3]))

        # send EOF
        await stream.send_message(api_pb2.TaskLogsBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxGetLogs", sandbox_get_logs)

        with enable_output():
            stdout: StreamReader[str] = make_stream_reader(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                object_id="sb-123",
                client=client,
                by_line=True,
            )

            out = []
            for line in stdout:
                out.append(line)

            assert out == ["foobar\n", "baz"]


@pytest.mark.asyncio
async def test_stream_reader_bytes_mode(servicer, client):
    """Test that the stream reader works in bytes mode."""

    class _BytesRouter:
        async def exec_stdio_read(self, task_id, exec_id, file_descriptor, deadline=None):
            yield sr_pb2.TaskExecStdioReadResponse(data=b"foo\n")

    router = _BytesRouter()
    with enable_output():
        stdout: StreamReader[bytes] = await _make_stream_reader.aio(
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
            object_id="tp-123",
            command_router_client=router,  # type: ignore[arg-type]
            task_id="task-1",
            text=False,
        )
        assert await stdout.read.aio() == b"foo\n"


def test_stream_reader_line_buffered_bytes(servicer, client):
    """Test that using line-buffering with bytes mode fails."""

    class _DummyRouter:
        pass

    with pytest.raises(ValueError):
        make_stream_reader(
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
            object_id="tp-123",
            command_router_client=_DummyRouter(),  # type: ignore[arg-type]
            task_id="task-1",
            by_line=True,
            text=False,
        )


@pytest.mark.asyncio
async def test_stream_reader_async_iter(servicer, client):
    """Test that StreamReader behaves as a proper async iterator."""

    async def sandbox_get_logs(servicer, stream):
        await stream.recv_message()

        log1 = api_pb2.TaskLogs(
            data="foo",
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
        )
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="0", items=[log1]))

        log2 = api_pb2.TaskLogs(
            data="bar",
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
        )
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="1", items=[log2]))

        # send EOF
        await stream.send_message(api_pb2.TaskLogsBatch(eof=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxGetLogs", sandbox_get_logs)

        expected = "foobar"

        stdout: StreamReader[str] = await _make_stream_reader.aio(
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
            object_id="sb-123",
            client=client,
            by_line=True,
        )

        out = ""
        async with aclosing(sync_or_async_iter(stdout)) as stream:
            async for line in stream:
                out += line

        assert out == expected


@pytest.mark.asyncio
async def test_stream_reader_container_process_reads_all_messages():
    """Test that StreamReader reads all messages from the command router."""

    class _MultiMsgRouter:
        async def exec_stdio_read(self, task_id, exec_id, file_descriptor, deadline=None):
            for i in range(6):
                yield sr_pb2.TaskExecStdioReadResponse(data=f"msg{i}\n".encode())

    router = _MultiMsgRouter()
    with enable_output():
        stdout: StreamReader[str] = await _make_stream_reader.aio(
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
            object_id="tp-123",
            command_router_client=router,  # type: ignore[arg-type]
            task_id="task-1",
            by_line=True,
        )

        output = []
        async for line in stdout:
            output.append(line)

        assert output == [f"msg{i}\n" for i in range(6)]


@pytest.mark.asyncio
async def test_stream_reader_timeout():
    """Test that StreamReader stops reading messages after the given deadline, and that
    messages are received within the deadline"""
    from modal.exception import ExecTimeoutError

    class _SlowRouter:
        async def exec_stdio_read(self, task_id, exec_id, file_descriptor, deadline=None):
            for i in range(3):
                if i == 2:
                    # Simulate the router raising a timeout error when deadline is exceeded
                    raise ExecTimeoutError("deadline exceeded")
                yield sr_pb2.TaskExecStdioReadResponse(data=f"msg{i}\n".encode())

    router = _SlowRouter()
    with enable_output():
        stdout: StreamReader[str] = await _make_stream_reader.aio(
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
            object_id="tp-123",
            command_router_client=router,  # type: ignore[arg-type]
            task_id="task-1",
            by_line=True,
            deadline=time.monotonic() + 0.5,
        )
        output: list[str] = []
        async for line in stdout:
            output.append(line)
        # message 3 should not be received, due to the timeout
        assert output == [f"msg{i}\n" for i in range(2)]


async def _bytes_stream(items: list[bytes]):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_stream_by_line_yields_nothing_with_empty_input_stream():
    stream = _stream_by_line(_bytes_stream([]))
    result = [chunk async for chunk in stream]
    assert result == []


@pytest.mark.asyncio
async def test_stream_by_line_yields_entire_chunk_without_newline_at_end():
    stream = _stream_by_line(_bytes_stream([b"hello world"]))
    result = [chunk async for chunk in stream]
    assert result == [b"hello world"]


@pytest.mark.asyncio
async def test_stream_by_line_splits_single_chunk_into_multiple_lines():
    stream = _stream_by_line(_bytes_stream([b"a\nb\nc\n"]))
    result = [chunk async for chunk in stream]
    assert result == [b"a\n", b"b\n", b"c\n"]


@pytest.mark.asyncio
async def test_stream_by_line_merges_chunks_until_newline_then_yields_one_line():
    # "ab" + "c\n" should yield "abc\n" as a single line
    stream = _stream_by_line(_bytes_stream([b"ab", b"c\n", b"de\n"]))
    result = [chunk async for chunk in stream]
    assert result == [b"abc\n", b"de\n"]


@pytest.mark.asyncio
async def test_stream_by_line_yields_leftover_without_newline_at_end():
    stream = _stream_by_line(_bytes_stream([b"line1\nline2"]))
    result = [chunk async for chunk in stream]
    assert result == [b"line1\n", b"line2"]


@pytest.mark.asyncio
async def test_stream_by_line_handles_consecutive_empty_lines():
    stream = _stream_by_line(_bytes_stream([b"\n\n", b"a\n", b"\n"]))
    result = [chunk async for chunk in stream]
    assert result == [b"\n", b"\n", b"a\n", b"\n"]


@pytest.mark.asyncio
async def test_stream_by_line_raises_assertion_error_for_non_bytes_items():
    async def _bad_stream():
        yield "not-bytes"  # type: ignore[misc]

    with pytest.raises(AssertionError):
        async with aclosing(_stream_by_line(_bad_stream())) as stream:
            async for _ in stream:
                pass


@pytest.mark.asyncio
async def test_decode_bytes_stream_to_str_yields_nothing_with_empty_input_stream():
    stream = _decode_bytes_stream_to_str(_bytes_stream([]))
    result = [chunk async for chunk in stream]
    assert result == []


@pytest.mark.asyncio
async def test_decode_bytes_stream_to_str_decodes_single_ascii_chunk():
    stream = _decode_bytes_stream_to_str(_bytes_stream([b"hello"]))
    result = [chunk async for chunk in stream]
    assert result == ["hello"]


@pytest.mark.asyncio
async def test_decode_bytes_stream_to_str_decodes_multiple_chunks_in_order():
    stream = _decode_bytes_stream_to_str(_bytes_stream([b"hello", b" ", b"world"]))
    result = [chunk async for chunk in stream]
    assert result == ["hello", " ", "world"]


@pytest.mark.asyncio
async def test_decode_bytes_stream_to_str_decodes_utf8_multibyte_characters_in_single_chunk():
    stream = _decode_bytes_stream_to_str(_bytes_stream(["café".encode("utf-8")]))
    result = [chunk async for chunk in stream]
    assert result == ["café"]


@pytest.mark.asyncio
async def test_decode_bytes_stream_to_str_handles_multibyte_split_across_chunks():
    # 'é' in UTF-8 is b"\xc3\xa9"; splitting across chunks should be decoded incrementally
    stream = _decode_bytes_stream_to_str(_bytes_stream([b"caf\xc3", b"\xa9"]))
    result = [chunk async for chunk in stream]
    assert result == ["caf", "é"]


class _TestV2SandboxRouter:
    """Test fixture for ``TaskCommandRouterClient`` for V2 sandbox stdio."""

    def __init__(self, frames: list[sr_pb2.SandboxStdioReadV2Response]):
        self._frames = frames
        self.calls: list[dict[str, object]] = []

    async def sandbox_stdio_read(
        self,
        task_id: str,
        file_descriptor: "api_pb2.FileDescriptor.ValueType",
    ) -> AsyncGenerator[sr_pb2.SandboxStdioReadV2Response, None]:
        self.calls.append({"task_id": task_id, "file_descriptor": file_descriptor})
        for frame in self._frames:
            yield frame


@synchronizer.wrap
async def _make_v2_stream_reader(
    router,
    *,
    sandbox_id: str = "sb-test",
    task_id: str = "ta-test",
    file_descriptor: "api_pb2.FileDescriptor.ValueType" = api_pb2.FILE_DESCRIPTOR_STDOUT,
    text: bool = False,
    by_line: bool = False,
) -> StreamReader:
    async def resolve_router() -> tuple[str, TaskCommandRouterClient]:
        return task_id, typing.cast(TaskCommandRouterClient, router)

    return StreamReader(
        _StreamReaderThroughSandboxCommandRouterParams(
            file_descriptor=file_descriptor,
            sandbox_id=sandbox_id,
            resolve_router=resolve_router,
        ),
        text=text,
        by_line=by_line,
    )


@pytest.mark.asyncio
async def test_v2_stream_reader_yields_all_chunks_in_order():
    router = _TestV2SandboxRouter(
        [
            sr_pb2.SandboxStdioReadV2Response(data=b"foo", starting_offset=0),
            sr_pb2.SandboxStdioReadV2Response(data=b"bar", starting_offset=3),
            sr_pb2.SandboxStdioReadV2Response(data=b"baz", starting_offset=6),
        ]
    )
    reader: StreamReader[bytes] = await _make_v2_stream_reader.aio(router, text=False)
    assert await reader.read.aio() == b"foobarbaz"
    assert router.calls == [{"task_id": "ta-test", "file_descriptor": api_pb2.FILE_DESCRIPTOR_STDOUT}]


@pytest.mark.asyncio
async def test_v2_stream_reader_decodes_text_across_chunks():
    # 'é' = b"\xc3\xa9", split across frames to test incremental UTF-8 decode.
    router = _TestV2SandboxRouter(
        [
            sr_pb2.SandboxStdioReadV2Response(data=b"caf\xc3", starting_offset=0),
            sr_pb2.SandboxStdioReadV2Response(data=b"\xa9!", starting_offset=4),
        ]
    )
    reader: StreamReader[str] = await _make_v2_stream_reader.aio(router, text=True)
    assert await reader.read.aio() == "café!"


@pytest.mark.asyncio
async def test_v2_stream_reader_yields_lines_when_by_line():
    router = _TestV2SandboxRouter(
        [
            sr_pb2.SandboxStdioReadV2Response(data=b"foo\nba", starting_offset=0),
            sr_pb2.SandboxStdioReadV2Response(data=b"r\nbaz", starting_offset=6),
        ]
    )
    reader: StreamReader[str] = await _make_v2_stream_reader.aio(router, text=True, by_line=True)
    out = []
    async for line in reader:
        out.append(line)
    assert out == ["foo\n", "bar\n", "baz"]


@pytest.mark.asyncio
async def test_v2_stream_reader_routes_stderr_file_descriptor():
    router = _TestV2SandboxRouter([sr_pb2.SandboxStdioReadV2Response(data=b"err", starting_offset=0)])
    reader: StreamReader[bytes] = await _make_v2_stream_reader.aio(
        router, file_descriptor=api_pb2.FILE_DESCRIPTOR_STDERR, text=False
    )
    assert await reader.read.aio() == b"err"
    assert router.calls == [{"task_id": "ta-test", "file_descriptor": api_pb2.FILE_DESCRIPTOR_STDERR}]


@pytest.mark.asyncio
async def test_v2_stream_reader_warns_on_silent_advance_first_chunk(caplog):
    router = _TestV2SandboxRouter(
        [
            sr_pb2.SandboxStdioReadV2Response(data=b"tail", starting_offset=1024),
        ]
    )
    reader: StreamReader[bytes] = await _make_v2_stream_reader.aio(router, sandbox_id="sb-evicted", text=False)
    with caplog.at_level("WARNING", logger="modal-client"):
        assert await reader.read.aio() == b"tail"
    assert any("sb-evicted" in rec.message and "dropped first 1024 bytes" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_v2_stream_reader_no_warning_when_starting_offset_zero(caplog):
    router = _TestV2SandboxRouter(
        [
            sr_pb2.SandboxStdioReadV2Response(data=b"ok", starting_offset=0),
        ]
    )
    reader: StreamReader[bytes] = await _make_v2_stream_reader.aio(router, text=False)
    with caplog.at_level("WARNING", logger="modal-client"):
        assert await reader.read.aio() == b"ok"
    assert not any("dropped first" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_v2_stream_reader_warns_only_on_first_chunk(caplog):
    # Subsequent chunks with non-zero starting_offset are normal continuation,
    # not silent advances; the warning must fire at most once.
    router = _TestV2SandboxRouter(
        [
            sr_pb2.SandboxStdioReadV2Response(data=b"a", starting_offset=128),
            sr_pb2.SandboxStdioReadV2Response(data=b"b", starting_offset=129),
            sr_pb2.SandboxStdioReadV2Response(data=b"c", starting_offset=130),
        ]
    )
    reader: StreamReader[bytes] = await _make_v2_stream_reader.aio(router, text=False)
    with caplog.at_level("WARNING", logger="modal-client"):
        assert await reader.read.aio() == b"abc"
    drop_warnings = [rec for rec in caplog.records if "dropped first" in rec.message]
    assert len(drop_warnings) == 1


@pytest.mark.asyncio
async def test_v2_stream_reader_raises_on_empty_data_frame():
    router = _TestV2SandboxRouter(
        [sr_pb2.SandboxStdioReadV2Response(data=b"", starting_offset=0)],
    )
    reader: StreamReader[bytes] = await _make_v2_stream_reader.aio(router, text=False)
    with pytest.raises(ValueError, match="Received empty message"):
        await reader.read.aio()


@pytest.mark.asyncio
async def test_v2_stream_reader_does_not_resolve_router_until_iterated():
    resolve_calls = 0

    async def resolve_router() -> tuple[str, TaskCommandRouterClient]:
        nonlocal resolve_calls
        resolve_calls += 1
        router = _TestV2SandboxRouter([sr_pb2.SandboxStdioReadV2Response(data=b"x", starting_offset=0)])
        return "ta-test", typing.cast(TaskCommandRouterClient, router)

    @synchronizer.wrap
    async def _build() -> StreamReader:
        return StreamReader(
            _StreamReaderThroughSandboxCommandRouterParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                sandbox_id="sb-test",
                resolve_router=resolve_router,
            ),
            text=False,
        )

    reader: StreamReader[bytes] = await _build.aio()  # type: ignore[attr-defined]
    assert resolve_calls == 0
    assert await reader.read.aio() == b"x"
    assert resolve_calls == 1


# ---------------------------------------
# _StreamWriter (command router) tests
# ---------------------------------------


class _FakeCommandRouterClient:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    async def exec_stdin_write(self, *, task_id: str, exec_id: str, offset: int, data: bytes, eof: bool) -> None:
        self.calls.append(
            {
                "task_id": task_id,
                "exec_id": exec_id,
                "offset": offset,
                "data": data,
                "eof": eof,
            }
        )

    async def exec_stdio_read(
        self,
        task_id: str,
        exec_id: str,
        file_descriptor: "api_pb2.FileDescriptor.ValueType",
        deadline: Optional[float] = None,
    ) -> AsyncGenerator[sr_pb2.TaskExecStdioReadResponse, None]:
        yield sr_pb2.TaskExecStdioReadResponse(data=b"a")
        yield sr_pb2.TaskExecStdioReadResponse(data=b"b")
        yield sr_pb2.TaskExecStdioReadResponse(data=b"c")


@pytest.mark.asyncio
async def test_stream_writer_drain_calls_exec_stdin_with_eof_when_closed_and_no_data():
    router = _FakeCommandRouterClient()
    writer = _StreamWriter(
        _StreamWriterThroughCommandRouterParams(
            task_id="task-1",
            object_id="tp-123",
            command_router_client=router,  # type: ignore[arg-type]
        )
    )

    writer.write_eof()
    await writer.drain()

    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["task_id"] == "task-1"
    assert call["exec_id"] == "tp-123"
    assert call["offset"] == 0
    assert call["data"] == b""
    assert call["eof"] is True


@pytest.mark.asyncio
async def test_stream_writer_drain_writes_all_written_data_since_last_drain():
    router = _FakeCommandRouterClient()
    writer = _StreamWriter(
        _StreamWriterThroughCommandRouterParams(
            task_id="task-1",
            object_id="tp-123",
            command_router_client=router,  # type: ignore[arg-type]
        )
    )

    writer.write(b"abc")
    writer.write(b"def")
    await writer.drain()

    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["offset"] == 0
    assert call["data"] == b"abcdef"
    assert call["eof"] is False


@pytest.mark.asyncio
async def test_stream_writer_drain_does_not_rewrite_data_written_prior_to_last_drain():
    router = _FakeCommandRouterClient()
    writer = _StreamWriter(
        _StreamWriterThroughCommandRouterParams(
            task_id="task-1",
            object_id="tp-123",
            command_router_client=router,  # type: ignore[arg-type]
        )
    )

    writer.write(b"ab")
    await writer.drain()

    writer.write(b"cd")
    await writer.drain()

    assert len(router.calls) == 2
    first, second = router.calls
    assert first["offset"] == 0
    assert first["data"] == b"ab"
    assert first["eof"] is False

    assert second["offset"] == 2  # advances by len(b"ab")
    assert second["data"] == b"cd"
    assert second["eof"] is False


@pytest.mark.asyncio
async def test_stream_writer_drain_with_data_and_eof_calls_exec_stdin_write_with_both():
    router = _FakeCommandRouterClient()
    writer = _StreamWriter(
        _StreamWriterThroughCommandRouterParams(
            task_id="task-1",
            object_id="tp-123",
            command_router_client=router,  # type: ignore[arg-type]
        )
    )

    writer.write(b"xyz")
    writer.write_eof()
    await writer.drain()

    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["offset"] == 0
    assert call["data"] == b"xyz"
    assert call["eof"] is True


@pytest.mark.parametrize("text, expected_out", [(False, b"abc"), (True, "abc")])
def test_stream_reader_read_concatenates_chunks(text, expected_out):
    router = _FakeCommandRouterClient()
    reader: typing.Union[StreamReader[str], StreamReader[bytes]] = make_stream_reader(
        file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
        object_id="tp-123",
        command_router_client=router,  # type: ignore[arg-type]
        task_id="task-1",
        text=text,
    )

    out = reader.read()
    assert out == expected_out
