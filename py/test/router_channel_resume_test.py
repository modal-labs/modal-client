# Copyright Modal Labs 2025
"""End-to-end checks that a released Sandbox channel resumes stdio at the right offset.

These run against a real grpclib server over a real socket, so they exercise what
happens to a suspended stdio stream when its connection is actually torn down.
"""

import asyncio
import gc
import pytest
import time

import psutil
import pytest_asyncio
from grpclib import GRPCError, Status
from grpclib.health.service import Health
from grpclib.server import Server

from modal._utils.grpc_utils import ModalChannel, create_channel_config
from modal._utils.task_command_router_client import TaskCommandRouterClient
from modal.exception import ClientClosed
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .conftest import MockTaskCommandRouterServicer

# The client API names descriptors with the api_pb2 enum; the wire uses the
# task-command-router one, which numbers them differently.
STDOUT = api_pb2.FILE_DESCRIPTOR_STDOUT
STDERR = api_pb2.FILE_DESCRIPTOR_STDERR
_WIRE_TO_FD = {
    sr_pb2.TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDOUT: STDOUT,
    sr_pb2.TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDERR: STDERR,
}

# Output the fake worker holds for the exec, served from any requested offset.
# The two descriptors differ in length so a stream resuming at the wrong one is visible.
# Short, because these tests wait the countdown out on the real clock.
SHORT_IDLE_TIMEOUT = 0.05

OUTPUT = {
    STDOUT: b"".join(f"line-{i}\n".encode() for i in range(10)),
    STDERR: b"".join(f"e{i}\n".encode() for i in range(10)),
}


class OffsetStdioServicer(MockTaskCommandRouterServicer):
    """Serves exec stdio from a byte offset, the way the worker seeks its stdio file."""

    def __init__(self):
        super().__init__()
        self.requested_offsets: dict[int, list[int]] = {STDOUT: [], STDERR: []}
        # Different per descriptor, so a stream that resumed at the other one's
        # offset would land somewhere visibly wrong.
        self.chunk_sizes = {STDOUT: 7, STDERR: 5}
        # Set to fail reads below a floor, standing in for an evicted rotating log.
        self.oldest_retained: int | None = None

    def offsets(self, fd=STDOUT) -> list[int]:
        return self.requested_offsets[fd]

    async def TaskExecStdioRead(self, stream) -> None:
        request: sr_pb2.TaskExecStdioReadRequest = await stream.recv_message()
        fd = _WIRE_TO_FD[request.file_descriptor]
        output = OUTPUT[fd]
        offset = int(request.offset)
        self.requested_offsets[fd].append(offset)
        if self.oldest_retained is not None and offset < self.oldest_retained:
            raise GRPCError(
                Status.OUT_OF_RANGE,
                f"offset {offset} has been evicted. Oldest retained offset is {self.oldest_retained}.",
            )
        while offset < len(output):
            chunk = output[offset : offset + self.chunk_sizes[fd]]
            await stream.send_message(sr_pb2.TaskExecStdioReadResponse(data=chunk))
            offset += len(chunk)
        # Hold the stream open so the test controls when the read ends.
        await asyncio.sleep(30)


class RouterEnv:
    """A local stdio server plus however many channels a test points at it."""

    def __init__(self, servicer: OffsetStdioServicer, port: int):
        self.servicer = servicer
        self.port = port
        self.clients: list[TaskCommandRouterClient] = []

    async def make_client(
        self, task_id: str = "ta-1", idle_timeout: float = SHORT_IDLE_TIMEOUT
    ) -> TaskCommandRouterClient:
        channel = ModalChannel(
            "127.0.0.1",
            self.port,
            ssl=None,
            config=create_channel_config(),
            closed_error_message="Unable to perform operation on a detached sandbox",
        )
        client = TaskCommandRouterClient(
            None,
            task_id,
            f"http://127.0.0.1:{self.port}",
            "jwt",
            channel,
            asyncio.get_running_loop(),
            asyncio.Lock(),
            stream_stdio_retry_delay_secs=0.01,
        )
        client.idle_timeout_secs = idle_timeout
        # Arm as `_connect` does after construction.
        client._arm_idle_release()
        self.clients.append(client)
        return client

    def open_sockets(self) -> int:
        """Connections this process holds to the server, as the OS sees them."""
        return sum(
            1
            for conn in psutil.Process().net_connections(kind="tcp")
            if conn.raddr and conn.raddr.port == self.port and conn.status == psutil.CONN_ESTABLISHED
        )

    async def settle(self) -> None:
        """Let asyncio finish closing sockets, which it does on a later loop turn."""
        for _ in range(5):
            await asyncio.sleep(0)


@pytest_asyncio.fixture
async def env():
    servicer = OffsetStdioServicer()
    server = Server([servicer, Health()])
    await server.start("127.0.0.1", 0)
    port = server._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]

    router_env = RouterEnv(servicer, port)
    yield router_env

    for client in router_env.clients:
        await client.close()
    server.close()
    await server.wait_closed()


@pytest_asyncio.fixture
async def router(env):
    """A single connected client, for tests that only need one."""
    yield await env.make_client(), env.servicer, env.port


def read_stdio(client, fd=STDOUT):
    return client.exec_stdio_read("ta-1", "exec-1", fd)


async def wait_released(client, timeout: float = 2.0) -> bool:
    """Whether the countdown gives the client's socket back within `timeout`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not client._channel._connected:
            return True
        await asyncio.sleep(0.005)
    return not client._channel._connected


async def stays_connected(client, duration: float | None = None) -> bool:
    """Whether the socket survives long enough for a countdown to have fired."""
    await asyncio.sleep(duration if duration is not None else client.idle_timeout_secs * 4)
    return client._channel._connected


@pytest.mark.asyncio
async def test_partly_consumed_stream_does_not_pin_the_channel(router):
    """Reading one chunk and stopping must leave the channel releasable."""
    client, _, _port = router
    stream = read_stdio(client)
    first = await anext(stream)
    assert first.data == OUTPUT[STDOUT][: len(first.data)]

    # The iterator is still referenced, so nothing has finalized it — yet the
    # caller holds no lease while it sits on the chunk.
    assert client._inflight == 0
    assert await wait_released(client)
    assert client._channel._connected is False

    await stream.aclose()


@pytest.mark.asyncio
async def test_released_channel_resumes_from_the_same_offset(router):
    """A reader that comes back after a release must see the next byte, not a gap."""
    client, servicer, _port = router
    stream = read_stdio(client)

    received = bytearray()
    received += (await anext(stream)).data
    consumed = len(received)

    # The socket goes away while the consumer holds the chunk.
    assert await wait_released(client)

    async for item in stream:
        received += item.data
        if len(received) >= len(OUTPUT[STDOUT]):
            break
    await stream.aclose()

    assert bytes(received) == OUTPUT[STDOUT]
    # Reopened exactly where the consumer had got to.
    assert servicer.offsets() == [0, consumed]


@pytest.mark.asyncio
async def test_resume_does_not_spend_the_retry_budget(router):
    """A deliberate release is not a failure, so it must not consume retries."""
    client, servicer, _port = router
    client.stream_stdio_max_retries = 1
    stream = read_stdio(client)

    received = bytearray()
    received += (await anext(stream)).data
    # More releases than the retry budget would tolerate if they counted as errors.
    for _ in range(4):
        assert await wait_released(client)
        received += (await anext(stream)).data

    await stream.aclose()
    assert bytes(received) == OUTPUT[STDOUT][: len(received)]
    assert len(servicer.offsets()) == 5


@pytest.mark.asyncio
async def test_output_lost_while_suspended_raises(router):
    """When the worker can no longer serve the offset, the reader is told."""
    client, servicer, _port = router
    stream = read_stdio(client)

    await anext(stream)
    assert await wait_released(client)
    # The output the consumer was part-way through is gone.
    servicer.oldest_retained = len(OUTPUT[STDOUT])

    with pytest.raises(Exception) as exc_info:
        async for _ in stream:
            pass
    assert "evicted" in str(exc_info.value)
    await stream.aclose()


@pytest.mark.asyncio
async def test_release_frees_the_socket_without_relying_on_gc(env, router):
    """The socket goes back to the OS when the channel is released, rather than
    whenever the interpreter next collects the stream."""
    client, servicer, _port = router
    gc.disable()
    try:
        stream = read_stdio(client)
        received = bytearray()
        received += (await anext(stream)).data
        consumed = len(received)
        assert env.open_sockets() == 1

        assert await wait_released(client)
        await env.settle()

        # `stream` is still strongly referenced and nothing has been collected.
        assert stream is not None
        assert env.open_sockets() == 0

        # Picking the stream back up reconnects and continues where it left off.
        received += (await anext(stream)).data
        assert env.open_sockets() == 1
        assert servicer.offsets() == [0, consumed]
        assert bytes(received) == OUTPUT[STDOUT][: len(received)]

        await stream.aclose()
    finally:
        gc.enable()


# ---------------------------------------------------------------------------
# A long-lived process holding many Sandboxes must not accumulate connections,
# whether or not it ever finishes reading their output.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_many_partly_read_sandboxes_are_all_released(env):
    """The shape of an agent server: many Sandboxes held at once, output half-read."""
    gc.disable()
    try:
        # Long enough that setting all twenty-five up cannot outlast the first
        # one's countdown: they have to be open at once for this to mean
        # anything.
        clients = [await env.make_client(task_id=f"ta-{i}", idle_timeout=2.0) for i in range(25)]
        streams = []
        for client in clients:
            stream = read_stdio(client)
            await anext(stream)
            streams.append(stream)

        assert env.open_sockets() == 25
        assert all(c._inflight == 0 for c in clients)

        # Every Sandbox and every half-read stream is still referenced, so
        # nothing here is reclaimed by going out of scope.
        deadline = time.monotonic() + 5
        while env.open_sockets() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await env.settle()
        assert len(streams) == 25
        assert env.open_sockets() == 0, "idle connections were never released"

        for stream in streams:
            await stream.aclose()
    finally:
        gc.enable()


@pytest.mark.asyncio
async def test_repeated_create_and_abandon_does_not_accumulate(env):
    """Sandbox churn must not grow the connection count."""
    # Long enough that nothing here is released by falling idle: walking away is
    # what has to reclaim it.
    for i in range(20):
        client = await env.make_client(task_id=f"ta-{i}", idle_timeout=30.0)
        stream = read_stdio(client)
        await anext(stream)
        # Walk away from both the stream and the Sandbox, as a caller would.
        del stream
        await client.close()
        env.clients.remove(client)
        del client
        gc.collect()
        await env.settle()

        # Each iteration must land back where the last one started.
        assert env.open_sockets() == 0, f"leaked a connection on iteration {i}"


@pytest.mark.asyncio
async def test_the_countdown_releases_a_half_read_stream(env):
    """The countdown reclaims a half-read stream without anything else prompting it."""
    client = await env.make_client()
    stream = read_stdio(client)
    received = bytearray()
    received += (await anext(stream)).data
    consumed = len(received)

    deadline = asyncio.get_running_loop().time() + 5
    while env.open_sockets() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    assert env.open_sockets() == 0, "the countdown did not release the idle channel"

    # And the half-read stream still picks up where it left off.
    received += (await anext(stream)).data
    assert env.servicer.offsets() == [0, consumed]
    assert bytes(received) == OUTPUT[STDOUT][: len(received)]
    await stream.aclose()


# ---------------------------------------------------------------------------
# Resuming a partly-read stream against a live Sandbox.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stdout_and_stderr_resume_at_their_own_offsets(router):
    """One release takes the shared channel, so both streams must recover separately."""
    client, servicer, _port = router
    out_stream = read_stdio(client, STDOUT)
    err_stream = read_stdio(client, STDERR)

    out = bytearray((await anext(out_stream)).data)
    err = bytearray((await anext(err_stream)).data)
    out_consumed, err_consumed = len(out), len(err)
    assert out_consumed != err_consumed, "descriptors should differ, or this proves nothing"

    assert await wait_released(client)

    out += (await anext(out_stream)).data
    err += (await anext(err_stream)).data

    assert servicer.offsets(STDOUT) == [0, out_consumed]
    assert servicer.offsets(STDERR) == [0, err_consumed]
    assert bytes(out) == OUTPUT[STDOUT][: len(out)]
    assert bytes(err) == OUTPUT[STDERR][: len(err)]

    await out_stream.aclose()
    await err_stream.aclose()


@pytest.mark.asyncio
async def test_resume_after_other_work_reconnected_the_channel(router):
    """A reader must resume correctly even when something else reopened the socket."""
    client, servicer, _port = router
    stream = read_stdio(client)
    received = bytearray()
    received += (await anext(stream)).data
    consumed = len(received)

    assert await wait_released(client)
    # Unrelated work reconnects the channel while the reader is still suspended.
    await client.exec_stdin_status(task_id="ta-1", exec_id="exec-1")
    assert client._channel._connected is True

    received += (await anext(stream)).data
    assert servicer.offsets() == [0, consumed]
    assert bytes(received) == OUTPUT[STDOUT][: len(received)]
    await stream.aclose()


@pytest.mark.asyncio
async def test_stream_survives_many_release_resume_cycles(router):
    """Releasing between every chunk must still reassemble the output exactly."""
    client, servicer, _port = router
    stream = read_stdio(client)

    received = bytearray()
    async for item in stream:
        received += item.data
        if len(received) >= len(OUTPUT[STDOUT]):
            break
        assert await wait_released(client)
    await stream.aclose()

    assert bytes(received) == OUTPUT[STDOUT]
    # One open per chunk, each asking for exactly what the reader had consumed.
    chunk = servicer.chunk_sizes[STDOUT]
    assert servicer.offsets() == [chunk * i for i in range(len(servicer.offsets()))]


# ---------------------------------------------------------------------------
# Lease accounting must survive the ways a stream can end.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lease_balances_when_the_consumer_raises(router):
    """An exception in the consumer closes the generator; the lease must unwind with it."""
    client, _, _port = router
    stream = read_stdio(client)

    with pytest.raises(RuntimeError):
        async for _ in stream:
            raise RuntimeError("consumer blew up")
    await stream.aclose()

    assert client._inflight == 0
    assert await wait_released(client)


@pytest.mark.asyncio
async def test_lease_balances_when_closed_mid_stream(router):
    """Closing part-way through must neither leak a lease nor drop one twice."""
    client, _, _port = router
    stream = read_stdio(client)
    await anext(stream)
    await anext(stream)
    assert client._inflight == 0

    await stream.aclose()
    assert client._inflight == 0
    # A second close is a no-op rather than a second decrement.
    await stream.aclose()
    assert client._inflight == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("close_when_idle", [False, True])
async def test_resume_after_client_closed_raises(router, close_when_idle):
    """A Sandbox whose client has gone away must fail the read, not hang on it."""
    client, _, _port = router
    stream = read_stdio(client)
    await anext(stream)

    if close_when_idle:
        await client.close_when_idle()
    else:
        await client.close()

    with pytest.raises(ClientClosed):
        await anext(stream)
    await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_timeout", [0, 3600])
@pytest.mark.parametrize("cancel_rpc", [False, True])
async def test_close_when_idle_preserves_active_rpc(idle_timeout, cancel_rpc):
    started = asyncio.Event()
    finish = asyncio.Event()

    class WaitServicer(OffsetStdioServicer):
        async def TaskExecWait(self, stream):
            await stream.recv_message()
            started.set()
            await finish.wait()
            await stream.send_message(sr_pb2.TaskExecWaitResponse(code=42))

    servicer = WaitServicer()
    server = Server([servicer, Health()])
    await server.start("127.0.0.1", 0)
    port = server._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]
    env = RouterEnv(servicer, port)
    client = await env.make_client(idle_timeout=idle_timeout)
    waiting = asyncio.create_task(client.exec_wait("ta-1", "exec-1"))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        await client.close_when_idle()
        await env.settle()
        assert not waiting.done()
        assert client._channel._connected
        with pytest.raises(ClientClosed):
            await client.exec_wait("ta-1", "exec-2")

        if cancel_rpc:
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        else:
            finish.set()
            assert (await asyncio.wait_for(waiting, timeout=2)).code == 42
        await env.settle()
        assert not client._channel._connected
        assert client._inflight == 0
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_connection_is_released_one_idle_timeout_after_a_partial_read(env):
    """The setting has to mean about what it says.

    A Sandbox read once and then forgotten should give its connection up one idle
    timeout later, not some multiple of it.
    """
    budget = 1.0
    client = await env.make_client(idle_timeout=budget)
    stream = read_stdio(client)
    await anext(stream)
    assert env.open_sockets() == 1

    started = time.monotonic()
    deadline = started + 4 * budget
    while env.open_sockets() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    elapsed = time.monotonic() - started

    assert env.open_sockets() == 0, "the idle connection was never released"
    assert elapsed >= budget / 2, "released far sooner than the configured timeout"
    assert elapsed < 2 * budget, "took multiples of the configured timeout to release"

    await stream.aclose()


@pytest.mark.asyncio
async def test_a_briefly_slow_reader_keeps_its_connection(env):
    """A caller still working through a chunk should not lose their connection.

    A read gives the channel up while the caller has the chunk, so the idle
    timeout is what governs between chunks. A caller who comes back inside it
    carries on reading the same stream rather than paying to reopen.
    """
    idle = 0.5
    client = await env.make_client(idle_timeout=idle)

    stream = read_stdio(client)
    received = bytearray()
    received += (await anext(stream)).data

    # Nothing is on the wire between chunks, so the countdown is what governs.
    assert client._inflight == 0
    assert await stays_connected(client, idle / 2), "released while the caller was still within the idle timeout"

    # So a caller who comes back carries on reading the same stream.
    received += (await anext(stream)).data
    assert bytes(received) == OUTPUT[STDOUT][: len(received)]
    assert env.servicer.offsets() == [0], "the stream should not have reopened"

    # Past it the socket goes back, which is what the timeout was deferring.
    assert await wait_released(client)
    await stream.aclose()


@pytest.mark.asyncio
async def test_other_work_does_not_cut_a_readers_time_short(env):
    """An unrelated call finishing must not pull the deadline in under a reader.

    The reader is still mid-chunk; that another operation happened to end sooner
    says nothing about them.
    """
    idle = 0.5
    client = await env.make_client(idle_timeout=idle)

    stream = read_stdio(client)
    await anext(stream)

    # Something else uses the channel and finishes while the reader sits.
    await client.exec_stdin_status(task_id="ta-1", exec_id="exec-1")

    assert await stays_connected(client, idle / 2), "an unrelated call shortened the reader's time"
    assert await wait_released(client)
    await stream.aclose()


@pytest.mark.asyncio
async def test_a_reader_waiting_on_output_cannot_be_released_under(env):
    """A release can only land while a consumer holds a chunk, never mid-receive.

    That is what lets the stream check for one just after handing a chunk over,
    rather than inferring it from whatever the next receive throws. It holds
    because an attempt keeps its lease except across that hand-over - so if the
    lease is ever narrowed, this is the test that should say so.
    """
    client = await env.make_client(idle_timeout=0.01)

    stream = read_stdio(client)
    received = bytearray()

    async def drain() -> None:
        async for item in stream:
            received.extend(item.data)

    consumer = asyncio.create_task(drain())
    try:
        # Once the output is consumed the server holds the stream open, leaving
        # the reader waiting on a worker that has nothing more to say.
        deadline = time.monotonic() + 5
        while len(received) < len(OUTPUT[STDOUT]) and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert bytes(received) == OUTPUT[STDOUT]
        await asyncio.sleep(0.05)

        assert client._inflight == 1, "a reader waiting on output should hold the channel"
        assert await stays_connected(client), "released the channel under a reader that was still waiting on output"
        assert env.open_sockets() == 1
    finally:
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await stream.aclose()


@pytest.mark.asyncio
async def test_a_parked_read_fails_promptly_when_its_client_closes(env):
    """Closing the client - what a detach does - must fail a parked read, not
    leave it wedged on a stream whose connection is gone for good."""
    client = await env.make_client(idle_timeout=0)

    stream = read_stdio(client)
    received = bytearray()
    while len(received) < len(OUTPUT[STDOUT]):
        received += (await anext(stream)).data

    # The worker has nothing more to say, so this read parks on the wire.
    waiting = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.05)
    assert not waiting.done()

    await client.close()
    with pytest.raises(ClientClosed):
        await asyncio.wait_for(waiting, timeout=5)
    await stream.aclose()
