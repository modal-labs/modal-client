# Copyright Modal Labs 2025
"""
Unit tests for task command router client.
"""

import asyncio
import pytest
import time
from typing import List, Optional

from grpclib import GRPCError, Status
from grpclib.client import Channel
from grpclib.exceptions import StreamTerminatedError

from modal._utils.task_command_router_client import TaskCommandRouterClient
from modal.exception import ExecTimeoutError
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2


def _start_index_for_offset(pieces: List[bytes], offset: int) -> int:
    """Return the index of the piece that starts at the given byte offset."""
    total = 0
    for idx, p in enumerate(pieces):
        if total == offset:
            return idx
        total += len(p)
    if total == offset:
        return len(pieces)
    raise AssertionError(f"Offset {offset} does not align to piece boundary")


class _UnaryStreamMethod:
    def __init__(self, open_fn):
        self._open_fn = open_fn

    def open(self, timeout: Optional[float] = None):  # match grpclib signature
        return self._open_fn(timeout)


class _Stub:
    def __init__(self, open_fn):
        self.TaskExecStdioRead = _UnaryStreamMethod(open_fn)


def create_dummy_channel() -> Channel:
    return Channel("https://router.test", ssl=False)


@pytest.mark.asyncio
async def test_exec_stdio_read_streams_stdout_batches(monkeypatch):
    pieces = [b"hello ", b"world", b"!\n"]

    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
    )

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout
            self._idx = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._idx >= len(pieces):
                raise StopAsyncIteration
            data = pieces[self._idx]
            self._idx += 1
            return sr_pb2.TaskExecStdioReadResponse(data=data)

    def _open(timeout: Optional[float] = None):
        return _Stream(timeout)

    fake_stub = _Stub(_open)
    client._stub = fake_stub  # type: ignore[assignment]

    out: List[bytes] = []
    async for item in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
        out.append(item.data)

    assert out == pieces
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_auth_retry_resumes_from_correct_offset(monkeypatch):
    # 2 pieces, then disconnect; next attempt fails auth on send once; then resume with remaining pieces.
    pieces = [b"AA", b"BB", b"CC", b"DD"]

    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
    )

    num_attempts_made = 0

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout
            self._emitted = 0
            self._last_req: Optional[sr_pb2.TaskExecStdioReadRequest] = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            self._last_req = req
            if num_attempts_made == 2:
                raise GRPCError(Status.UNAUTHENTICATED, "auth required")

        def __aiter__(self):
            return self

        async def __anext__(self):
            assert self._last_req is not None
            start_idx = _start_index_for_offset(pieces, int(self._last_req.offset))
            # On first attempt, fail after 2 items to simulate stream drop. The attempt
            # after that will fail in send_message with UNAUTHENTICATED.
            if num_attempts_made == 1 and self._emitted >= 2:
                raise StreamTerminatedError("stream closed")
            next_idx = start_idx + self._emitted
            if next_idx >= len(pieces):
                raise StopAsyncIteration
            self._emitted += 1
            return sr_pb2.TaskExecStdioReadResponse(data=pieces[next_idx])

    def _open(timeout: Optional[float] = None):
        nonlocal num_attempts_made
        num_attempts_made += 1
        return _Stream(timeout)

    fake_stub = _Stub(_open)

    num_refreshes_made = 0

    async def _refresh_jwt():
        nonlocal num_refreshes_made
        num_refreshes_made += 1
        client._jwt = "t2"

    client._refresh_jwt = _refresh_jwt  # type: ignore[assignment]
    client._stub = fake_stub  # type: ignore[assignment]

    out: List[bytes] = []
    async for item in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
        out.append(item.data)

    assert out == pieces
    # We should have refreshed exactly once due to UNAUTHENTICATED on send.
    assert num_refreshes_made == 1
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_transient_error_retry_resumes_from_correct_offset(monkeypatch):
    # 2 pieces, then transient error; then resume with remaining pieces.
    pieces = [b"one", b"two", b"three"]

    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
    )

    num_attempts_made = 0

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout
            self._emitted = 0
            self._last_req: Optional[sr_pb2.TaskExecStdioReadRequest] = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            self._last_req = req

        def __aiter__(self):
            return self

        async def __anext__(self):
            assert self._last_req is not None
            start_idx = _start_index_for_offset(pieces, int(self._last_req.offset))
            if num_attempts_made == 1 and self._emitted >= 2:
                raise GRPCError(Status.UNAVAILABLE, "unavailable")
            next_idx = start_idx + self._emitted
            if next_idx >= len(pieces):
                raise StopAsyncIteration
            self._emitted += 1
            return sr_pb2.TaskExecStdioReadResponse(data=pieces[next_idx])

    def _open(timeout: Optional[float] = None):
        nonlocal num_attempts_made
        num_attempts_made += 1
        return _Stream(timeout)

    client._stub = _Stub(_open)  # type: ignore[assignment]

    out: List[bytes] = []
    async for item in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
        out.append(item.data)

    assert out == pieces
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_auth_fails_twice_raises_auth_error(monkeypatch):
    pieces = [b"x", b"y"]

    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
    )

    num_attempts_made = 0

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout
            self._last_req: Optional[sr_pb2.TaskExecStdioReadRequest] = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            self._last_req = req
            if num_attempts_made in (1, 2):
                raise GRPCError(Status.UNAUTHENTICATED, "auth required")

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    def _open(timeout: Optional[float] = None):
        nonlocal num_attempts_made
        num_attempts_made += 1
        return _Stream(timeout)

    num_refreshes_made = 0

    async def _refresh_jwt():
        nonlocal num_refreshes_made
        num_refreshes_made += 1
        client._jwt = f"t{num_refreshes_made}"

    client._refresh_jwt = _refresh_jwt  # type: ignore[assignment]
    client._stub = _Stub(_open)  # type: ignore[assignment]

    with pytest.raises(GRPCError) as exc_info:
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
            pass
    assert exc_info.value.status == Status.UNAUTHENTICATED
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_unavailable_forever_raises_grpcerror(monkeypatch):
    # Simulate UNAVAILABLE on every attempt immediately upon starting iteration.
    pieces = [b"irrelevant"]

    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
        stream_stdio_retry_delay=0.001,
        stream_stdio_retry_delay_factor=1.0,
        stream_stdio_max_retries=5,
    )

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise GRPCError(Status.UNAVAILABLE, "unavailable")

    def _open(timeout: Optional[float] = None):
        return _Stream(timeout)

    client._stub = _Stub(_open)  # type: ignore[assignment]

    with pytest.raises(GRPCError) as exc_info:
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
            pass

    assert exc_info.value.status == Status.UNAVAILABLE
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_open_raises_once_then_succeeds(monkeypatch):
    pieces = [b"data"]

    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
    )

    raised = False

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout
            self._emitted = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._emitted:
                raise StopAsyncIteration
            self._emitted = True
            return sr_pb2.TaskExecStdioReadResponse(data=pieces[0])

    def _open(timeout: Optional[float] = None):
        nonlocal raised
        if not raised:
            raised = True
            raise GRPCError(Status.UNAVAILABLE, "open failed")
        return _Stream(timeout)

    client._stub = _Stub(_open)  # type: ignore[assignment]

    out: List[bytes] = []
    async for item in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
        out.append(item.data)
    assert out == pieces
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_deadline_exceeded_on_send_raises_exec_timeout_error(monkeypatch):
    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
    )

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            assert self._timeout is not None
            await asyncio.sleep(self._timeout + 0.05)
            raise asyncio.TimeoutError()

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    def _open(timeout: Optional[float] = None):
        return _Stream(timeout)

    client._stub = _Stub(_open)  # type: ignore[assignment]

    deadline = time.monotonic() + 0.1
    with pytest.raises(ExecTimeoutError):
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT, deadline=deadline):
            pass
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_deadline_exceeded_on_first_item_raises_exec_timeout_error(monkeypatch):
    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
    )

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout
            self._sent_first = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._sent_first:
                self._sent_first = True
                assert self._timeout is not None
                await asyncio.sleep(self._timeout + 0.05)
                raise asyncio.TimeoutError()
            raise StopAsyncIteration

    def _open(timeout: Optional[float] = None):
        return _Stream(timeout)

    client._stub = _Stub(_open)  # type: ignore[assignment]

    deadline = time.monotonic() + 0.1
    with pytest.raises(ExecTimeoutError):
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT, deadline=deadline):
            pass
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_deadline_respected_on_attribute_error(monkeypatch):
    # AttributeError("_write_appdata") branch should respect deadline and raise
    # ExecTimeoutError instead of sleeping past it.
    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
        stream_stdio_retry_delay=0.2,  # longer than remaining time
    )

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise AttributeError("_write_appdata")

    def _open(timeout: Optional[float] = None):
        return _Stream(timeout)

    client._stub = _Stub(_open)  # type: ignore[assignment]

    # Set deadline less than retry delay so a retry would exceed it
    deadline = time.monotonic() + 0.05
    start = time.monotonic()
    with pytest.raises(ExecTimeoutError):
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT, deadline=deadline):
            pass
    elapsed = time.monotonic() - start
    # Should not sleep the full retry delay (0.2s); expect prompt timeout (< 0.15s).
    assert elapsed < 0.15
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_deadline_respected_on_stream_terminated_error(monkeypatch):
    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
        stream_stdio_retry_delay=0.2,
    )

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StreamTerminatedError("closed")

    def _open(timeout: Optional[float] = None):
        return _Stream(timeout)

    client._stub = _Stub(_open)  # type: ignore[assignment]

    deadline = time.monotonic() + 0.05
    start = time.monotonic()
    with pytest.raises(ExecTimeoutError):
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT, deadline=deadline):
            pass
    elapsed = time.monotonic() - start
    assert elapsed < 0.15
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_deadline_respected_on_oserror(monkeypatch):
    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
        stream_stdio_retry_delay=0.2,
    )

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise OSError("network down")

    def _open(timeout: Optional[float] = None):
        return _Stream(timeout)

    client._stub = _Stub(_open)  # type: ignore[assignment]

    deadline = time.monotonic() + 0.05
    start = time.monotonic()
    with pytest.raises(ExecTimeoutError):
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT, deadline=deadline):
            pass
    elapsed = time.monotonic() - start
    assert elapsed < 0.15
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_deadline_exceeded_on_open_raises_exec_timeout_error(monkeypatch):
    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        channel=create_dummy_channel(),
    )

    class _Stream:
        def __init__(self, timeout: Optional[float]):
            self._timeout = timeout

        async def __aenter__(self):
            # Simulate grpclib honoring per-RPC timeout during open
            assert self._timeout is not None
            await asyncio.sleep(self._timeout + 0.05)
            raise asyncio.TimeoutError()

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
            return False

        async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    def _open(timeout: Optional[float] = None):
        return _Stream(timeout)

    client._stub = _Stub(_open)  # type: ignore[assignment]

    # Set a small deadline; expect prompt ExecTimeoutError.
    deadline = time.monotonic() + 0.05
    start = time.monotonic()
    with pytest.raises(ExecTimeoutError):
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT, deadline=deadline):
            pass
    elapsed = time.monotonic() - start
    # Should not significantly exceed the deadline (allow small overhead)
    assert elapsed < 0.2
    await client.close()
