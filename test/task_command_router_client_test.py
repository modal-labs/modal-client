# Copyright Modal Labs 2025
"""
Unit tests for task command router client.
"""

import asyncio
import pytest
import time
from typing import List, Mapping, Optional, Tuple

from grpclib import GRPCError, Status
from grpclib.exceptions import StreamTerminatedError

from modal._utils.task_command_router_client import TaskCommandRouterClient
from modal.exception import ExecTimeoutError
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2


class _FakeStreamController:
    """Controls behavior of fake gRPC stream across attempts.

    This controller simulates server-side behavior for TaskExecStdioRead.
    It can inject authentication failures on send, and transient errors during iteration,
    while ensuring resumed reads start from the provided offset.
    """

    def __init__(
        self,
        pieces: List[bytes],
        *,
        auth_error_on_send_attempts: Optional[set[int]] = None,
        iter_error_plan: Optional[Mapping[int, Tuple[int, BaseException]]] = None,
        timeout_on_send_attempts: Optional[set[int]] = None,
        timeout_on_first_iter_attempts: Optional[set[int]] = None,
    ) -> None:
        self._pieces = pieces
        self._attempt = 0
        self._last_req: Optional[sr_pb2.TaskExecStdioReadRequest] = None
        self._auth_error_on_send_attempts = auth_error_on_send_attempts or set()
        # Map attempt -> (after_n_pieces_emitted_in_attempt, exception)
        self._iter_error_plan = iter_error_plan or {}
        self._timeout_on_send_attempts = timeout_on_send_attempts or set()
        self._timeout_on_first_iter_attempts = timeout_on_first_iter_attempts or set()

    def open(self):
        self._attempt += 1
        # A new FakeStream per open()
        return _FakeStream(self)

    async def on_send_message(self, req: sr_pb2.TaskExecStdioReadRequest) -> None:
        # Possibly fail authentication for this attempt on initial send
        if self._attempt in self._auth_error_on_send_attempts:
            # Only fail once for this attempt
            self._auth_error_on_send_attempts.remove(self._attempt)
            raise GRPCError(Status.UNAUTHENTICATED, "auth required")
        self._last_req = req

    def _start_index_for_offset(self, offset: int) -> int:
        """Return piece index to start from, assuming offset aligns to piece boundaries."""
        total = 0
        for idx, p in enumerate(self._pieces):
            if total == offset:
                return idx
            total += len(p)
        if total == offset:
            # Exactly at end
            return len(self._pieces)
        raise AssertionError(f"Offset {offset} does not align to piece boundary")

    async def next_item(self, emitted_in_attempt: int):
        assert self._last_req is not None, "send_message must be called before iterating"
        start_idx = self._start_index_for_offset(int(self._last_req.offset))

        # Induce an iteration error after N pieces for this attempt
        if self._attempt in self._iter_error_plan:
            after_n, exc = self._iter_error_plan[self._attempt]
            if emitted_in_attempt >= after_n:
                raise exc

        # Simulate timeout before first item for this attempt
        if self._attempt in self._timeout_on_first_iter_attempts and emitted_in_attempt == 0:
            self._timeout_on_first_iter_attempts.remove(self._attempt)
            raise asyncio.TimeoutError()

        next_idx = start_idx + emitted_in_attempt
        if next_idx >= len(self._pieces):
            raise StopAsyncIteration
        data = self._pieces[next_idx]
        return sr_pb2.TaskExecStdioReadResponse(data=data)


class _FakeUnaryStreamMethod:
    def __init__(self, controller: _FakeStreamController):
        self._controller = controller

    def open(self, timeout: Optional[float] = None):  # match grpclib signature
        stream = self._controller.open()
        stream._timeout = timeout  # type: ignore[attr-defined]
        return stream


class _FakeStub:
    def __init__(self, controller: _FakeStreamController):
        self.TaskExecStdioRead = _FakeUnaryStreamMethod(controller)


class _FakeStream:
    def __init__(self, controller: _FakeStreamController):
        self._controller = controller
        self._emitted_in_attempt = 0
        self._timeout: Optional[float] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201 - test helper
        return False

    async def send_message(self, req: sr_pb2.TaskExecStdioReadRequest, end: bool = True):  # noqa: ARG002
        # Simulate timeout on send for this attempt by sleeping beyond the timeout and then raising
        if self._controller._attempt in self._controller._timeout_on_send_attempts:
            # ensure one-time behavior per attempt
            self._controller._timeout_on_send_attempts.remove(self._controller._attempt)
            await asyncio.sleep((self._timeout or 0.2) + 0.05)
            raise asyncio.TimeoutError()
        # Otherwise proceed normally
        await self._controller.on_send_message(req)

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Simulate timeout before first item: sleep beyond timeout then raise
        if (
            self._emitted_in_attempt == 0
            and self._controller._attempt in self._controller._timeout_on_first_iter_attempts
        ):
            self._controller._timeout_on_first_iter_attempts.remove(self._controller._attempt)
            await asyncio.sleep((self._timeout or 0.2) + 0.05)
            raise asyncio.TimeoutError()

        try:
            item = await self._controller.next_item(self._emitted_in_attempt)
        except StopAsyncIteration:
            raise StopAsyncIteration
        self._emitted_in_attempt += 1
        return item


@pytest.mark.asyncio
async def test_exec_stdio_read_streams_stdout_batches(monkeypatch):
    pieces = [b"hello ", b"world", b"!\n"]

    # Create client and stub fakes
    client = TaskCommandRouterClient(server_client=None, task_id="sb-1", server_url="https://router.test", jwt="t")

    controller = _FakeStreamController(pieces)
    fake_stub = _FakeStub(controller)

    async def _noop_ensure_connected():
        return None

    async def _get_stub():
        return fake_stub

    # Patch instance methods to avoid real network
    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

    out: List[bytes] = []
    async for item in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
        out.append(item.data)

    assert out == pieces


@pytest.mark.asyncio
async def test_exec_stdio_read_auth_retry_resumes_from_correct_offset(monkeypatch):
    # 2 pieces, then disconnect; next attempt fails auth on send once; then resume with remaining pieces
    pieces = [b"AA", b"BB", b"CC", b"DD"]

    client = TaskCommandRouterClient(server_client=None, task_id="sb-1", server_url="https://router.test", jwt="t")

    controller = _FakeStreamController(
        pieces,
        auth_error_on_send_attempts={2},
        iter_error_plan={1: (2, StreamTerminatedError("stream closed"))},
    )
    fake_stub = _FakeStub(controller)

    async def _noop_ensure_connected():
        return None

    refresh_count = {"n": 0}

    async def _refresh_jwt():
        refresh_count["n"] += 1
        client._jwt = "t2"

    async def _get_stub():
        return fake_stub

    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._refresh_jwt = _refresh_jwt  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

    out: List[bytes] = []
    async for item in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
        out.append(item.data)

    assert out == pieces
    # We should have refreshed exactly once due to UNAUTHENTICATED on send
    assert refresh_count["n"] == 1


@pytest.mark.asyncio
async def test_exec_stdio_read_transient_error_retry_resumes_from_correct_offset(monkeypatch):
    # 2 pieces, then transient error; then resume with remaining pieces.
    pieces = [b"one", b"two", b"three"]

    client = TaskCommandRouterClient(server_client=None, task_id="sb-1", server_url="https://router.test", jwt="t")

    controller = _FakeStreamController(
        pieces,
        iter_error_plan={1: (2, GRPCError(Status.UNAVAILABLE, "unavailable"))},
    )
    fake_stub = _FakeStub(controller)

    async def _noop_ensure_connected():
        return None

    async def _get_stub():
        return fake_stub

    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

    out: List[bytes] = []
    async for item in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
        out.append(item.data)

    assert out == pieces


@pytest.mark.asyncio
async def test_exec_stdio_read_auth_fails_twice_raises_auth_error(monkeypatch):
    pieces = [b"x", b"y"]

    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
    )

    # Fail UNAUTHENTICATED on both first and second attempts  to call send_message).
    controller = _FakeStreamController(
        pieces,
        auth_error_on_send_attempts={1, 2},
    )
    fake_stub = _FakeStub(controller)

    async def _noop_ensure_connected():
        return None

    refresh_count = {"n": 0}

    async def _refresh_jwt():
        refresh_count["n"] += 1
        client._jwt = f"t{refresh_count['n']}"

    async def _get_stub():
        return fake_stub

    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._refresh_jwt = _refresh_jwt  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

    with pytest.raises(GRPCError) as exc_info:
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
            pass
    assert exc_info.value.status == Status.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_exec_stdio_read_unavailable_forever_raises_grpcerror(monkeypatch):
    # Simulate UNAVAILABLE on every attempt immediately upon starting iteration.
    pieces = [b"irrelevant"]

    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        stream_stdio_retry_delay=0.001,
        stream_stdio_retry_delay_factor=1.0,
        stream_stdio_max_retries=5,
    )

    # Prepare an iter_error_plan that forces UNAVAILABLE on many attempts.
    iter_error_plan = {attempt: (0, GRPCError(Status.UNAVAILABLE, "unavailable")) for attempt in range(1, 50)}
    controller = _FakeStreamController(pieces, iter_error_plan=iter_error_plan)
    fake_stub = _FakeStub(controller)

    async def _noop_ensure_connected():
        return None

    async def _get_stub():
        return fake_stub

    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

    with pytest.raises(GRPCError) as exc_info:
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
            pass

    assert exc_info.value.status == Status.UNAVAILABLE


@pytest.mark.asyncio
async def test_exec_stdio_read_open_raises_once_then_succeeds(monkeypatch):
    # This test expects a retry after open() raises once, and thus success.
    pieces = [b"data"]

    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
    )

    controller = _FakeStreamController(pieces)

    class _FlakyUnaryStreamMethod(_FakeUnaryStreamMethod):
        def __init__(self, controller: _FakeStreamController):
            super().__init__(controller)
            self._raised = False

        def open(self, timeout: Optional[float] = None):  # match grpclib signature
            if not self._raised:
                self._raised = True
                raise GRPCError(Status.UNAVAILABLE, "open failed")
            return super().open(timeout)

    class _FlakyStub(_FakeStub):
        def __init__(self, controller: _FakeStreamController):
            self.TaskExecStdioRead = _FlakyUnaryStreamMethod(controller)

    flaky_stub = _FlakyStub(controller)

    async def _noop_ensure_connected():
        return None

    async def _get_stub():
        return flaky_stub

    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

    out: List[bytes] = []
    async for item in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT):
        out.append(item.data)
    assert out == pieces
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_deadline_exceeded_on_send_raises_exec_timeout_error(monkeypatch):
    # Configure client with small retry windows; set deadline ~200ms from now
    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
    )

    controller = _FakeStreamController(pieces=[b"x"], timeout_on_send_attempts={1})
    fake_stub = _FakeStub(controller)

    async def _noop_ensure_connected():
        return None

    async def _get_stub():
        return fake_stub

    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

    # Set a near deadline; exec_stdio_read will compute timeout for open()
    deadline = time.monotonic() + 0.2
    with pytest.raises(ExecTimeoutError):
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT, deadline=deadline):
            pass


@pytest.mark.asyncio
async def test_exec_stdio_read_deadline_exceeded_on_first_item_raises_exec_timeout_error(monkeypatch):
    # Configure client with small retry windows; set deadline ~200ms from now
    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
    )

    controller = _FakeStreamController(pieces=[b"hello"], timeout_on_first_iter_attempts={1})
    fake_stub = _FakeStub(controller)

    async def _noop_ensure_connected():
        return None

    async def _get_stub():
        return fake_stub

    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

    deadline = time.monotonic() + 0.2
    with pytest.raises(ExecTimeoutError):
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT, deadline=deadline):
            pass


@pytest.mark.asyncio
async def test_exec_stdio_read_deadline_respected_on_attribute_error(monkeypatch):
    # AttributeError("_write_appdata") branch should respect deadline and raise
    # ExecTimeoutError instead of sleeping past it.
    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        stream_stdio_retry_delay=0.2,  # longer than remaining time
    )

    # Force AttributeError path immediately on iteration start
    pieces = [b"irrelevant"]
    controller = _FakeStreamController(pieces, iter_error_plan={1: (0, AttributeError("_write_appdata"))})
    fake_stub = _FakeStub(controller)

    async def _noop_ensure_connected():
        return None

    async def _get_stub():
        return fake_stub

    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

    # Set deadline less than retry delay so a retry would exceed it
    deadline = time.monotonic() + 0.05
    start = time.monotonic()
    with pytest.raises(ExecTimeoutError):
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT, deadline=deadline):
            pass
    elapsed = time.monotonic() - start
    # Should not sleep the full retry delay (0.2s); expect prompt timeout (< 0.15s)
    assert elapsed < 0.15
    await client.close()


@pytest.mark.asyncio
async def test_exec_stdio_read_deadline_respected_on_stream_terminated_error(monkeypatch):
    client = TaskCommandRouterClient(
        server_client=None,
        task_id="sb-1",
        server_url="https://router.test",
        jwt="t",
        stream_stdio_retry_delay=0.2,
    )

    pieces = [b"irrelevant"]
    controller = _FakeStreamController(pieces, iter_error_plan={1: (0, StreamTerminatedError("closed"))})
    fake_stub = _FakeStub(controller)

    async def _noop_ensure_connected():
        return None

    async def _get_stub():
        return fake_stub

    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

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
        stream_stdio_retry_delay=0.2,
    )

    class _OSErrorStream(_FakeStream):
        async def __anext__(self):
            raise OSError("network down")

    class _OSErrorController(_FakeStreamController):
        def open(self):
            self._attempt += 1
            return _OSErrorStream(self)

    controller = _OSErrorController([b"irrelevant"])  # pieces unused
    fake_stub = _FakeStub(controller)

    async def _noop_ensure_connected():
        return None

    async def _get_stub():
        return fake_stub

    client._ensure_connected = _noop_ensure_connected  # type: ignore[assignment]
    client._get_stub = _get_stub  # type: ignore[assignment]

    deadline = time.monotonic() + 0.05
    start = time.monotonic()
    with pytest.raises(ExecTimeoutError):
        async for _ in client.exec_stdio_read("task-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT, deadline=deadline):
            pass
    elapsed = time.monotonic() - start
    assert elapsed < 0.15
    await client.close()
