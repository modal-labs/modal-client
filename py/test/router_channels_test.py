# Copyright Modal Labs 2025
"""Unit tests for idle release of Sandbox connections and output streams."""

import asyncio
import pytest
import time

import pytest_asyncio

from modal._utils.grpc_utils import ModalChannel
from modal._utils.idle_countdown import IdleCountdown
from modal._utils.task_command_router_client import TaskCommandRouterClient
from modal.exception import ClientClosed
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .supports.skip import skip_windows

# Short, because these tests wait the countdown out on the real clock rather
# than handing the client a clock reading of their own.
IDLE_TIMEOUT = 0.05


async def released(client, timeout: float = 2.0) -> bool:
    """Whether the client releases its connection within `timeout`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client._channel.num_soft_closes:
            return True
        await asyncio.sleep(0.005)
    return bool(client._channel.num_soft_closes)


async def stays_connected(client, duration: float = IDLE_TIMEOUT * 4) -> bool:
    """Whether the connection remains open for `duration`."""
    await asyncio.sleep(duration)
    return client._channel.num_soft_closes == 0


class FakeChannel:
    """Stands in for a ModalChannel, recording how it was closed."""

    def __init__(self) -> None:
        self.connected = True
        self.num_soft_closes = 0
        self.num_permanent_closes = 0

    @property
    def _connected(self) -> bool:
        return self.connected

    def release_connection(self) -> None:
        self.num_soft_closes += 1
        self.connected = False

    def close(self) -> None:
        self.num_permanent_closes += 1
        self.connected = False


class UnaryStreamMethod:
    def __init__(self, open_fn):
        self._open_fn = open_fn

    def open(self, timeout: float | None = None, metadata: dict | None = None):  # noqa: ARG002
        return self._open_fn()


@pytest_asyncio.fixture
async def make_client():
    clients: list[TaskCommandRouterClient] = []

    async def _make(*, channel=None, idle_timeout: float = IDLE_TIMEOUT) -> TaskCommandRouterClient:
        client = TaskCommandRouterClient(
            None,
            "ta-1",
            "https://router.test",
            "jwt",
            channel or FakeChannel(),  # type: ignore[arg-type]
            asyncio.get_running_loop(),
            asyncio.Lock(),
        )
        client.idle_timeout_secs = idle_timeout
        # Arm as `_connect` does after construction.
        client._arm_idle_release()
        clients.append(client)
        return client

    yield _make

    for client in clients:
        await client.close()


def test_modal_channel_release_leaves_channel_usable():
    """Releasing a connection must not trip the terminal close guard."""
    channel = ModalChannel("router.test", 443, ssl=False, closed_error_message="detached")
    channel.release_connection()
    assert channel._permanently_closed is False

    channel.close()
    assert channel._permanently_closed is True


@pytest.mark.asyncio
async def test_idle_channel_is_released(make_client):
    client = await make_client()
    channel = client._channel

    assert await released(client)
    assert channel.num_soft_closes == 1
    assert channel.num_permanent_closes == 0


@pytest.mark.asyncio
async def test_released_channel_is_not_released_again(make_client):
    client = await make_client()
    assert await released(client)

    # Nothing re-arms after release, so the connection is released once.
    await asyncio.sleep(IDLE_TIMEOUT * 4)
    assert client._channel.num_soft_closes == 1


@pytest.mark.asyncio
async def test_in_flight_work_pins_the_channel(make_client):
    client = await make_client()

    async with client._lease():
        assert await stays_connected(client), "released while an operation held the channel"

    # Leaving the lease restarts the countdown from the full timeout.
    assert client._channel.num_soft_closes == 0
    assert await released(client)


@pytest.mark.asyncio
async def test_each_operation_restarts_the_countdown(make_client):
    """Work arriving inside the timeout keeps the connection open."""
    client = await make_client()

    # Spans several timeouts in total, so an unarmed countdown would have fired.
    for _ in range(6):
        async with client._lease():
            pass
        await asyncio.sleep(IDLE_TIMEOUT / 2)

    assert client._channel.num_soft_closes == 0
    assert await released(client)


@pytest.mark.asyncio
async def test_zero_timeout_disables_release(make_client):
    client = await make_client(idle_timeout=0)
    assert await stays_connected(client)


@pytest.mark.asyncio
async def test_closed_channel_rejects_further_use(make_client):
    client = await make_client()
    await client.close()

    assert client._channel.num_permanent_closes == 1
    with pytest.raises(ClientClosed):
        async with client._lease():
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("detach", [False, True])
async def test_stdio_stream_waiting_on_output_pins_the_channel(make_client, detach):
    """A consumer blocked inside `anext` keeps the connection in use."""
    client = await make_client()
    keep_open = asyncio.Event()

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send_message(self, req, end=True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            await keep_open.wait()
            raise StopAsyncIteration

    class Stub:
        TaskExecStdioRead = UnaryStreamMethod(lambda: Stream())

    client._stub = Stub()  # type: ignore[assignment]

    stream = client.exec_stdio_read("ta-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT)
    consumer = asyncio.create_task(anext(stream, None))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert client._inflight == 1
    if detach:
        await client.close_when_idle()
        assert client._channel.num_permanent_closes == 0
    assert await stays_connected(client), "released while a consumer was waiting for output"

    # Once the stream ends and the consumer detaches, the channel is releasable.
    keep_open.set()
    await consumer
    await stream.aclose()
    assert client._inflight == 0
    if detach:
        await asyncio.sleep(0.01)
        assert client._channel.num_permanent_closes == 1
    else:
        assert await released(client)


@pytest.mark.asyncio
async def test_partly_consumed_stdio_stream_holds_no_lease(make_client):
    """A caller sitting on a chunk is not using the channel, so it must not pin one."""
    client = await make_client()

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send_message(self, req, end=True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            return sr_pb2.TaskExecStdioReadResponse(data=b"chunk")

    class Stub:
        TaskExecStdioRead = UnaryStreamMethod(lambda: Stream())

    client._stub = Stub()  # type: ignore[assignment]

    stream = client.exec_stdio_read("ta-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT)
    assert (await anext(stream)).data == b"chunk"
    assert client._inflight == 0
    assert await released(client)

    # Closing leaves the lease count balanced.
    await stream.aclose()
    assert client._inflight == 0


@pytest.mark.asyncio
async def test_resume_after_client_close_does_not_read_buffered_output(make_client):
    """A terminal close is observed before resuming the stale stream."""
    client = await make_client(idle_timeout=3600)
    client.stream_stdio_max_retries = 0
    num_reads = 0

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send_message(self, req, end=True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            nonlocal num_reads
            num_reads += 1
            return sr_pb2.TaskExecStdioReadResponse(data=f"chunk-{num_reads}".encode())

    class Stub:
        TaskExecStdioRead = UnaryStreamMethod(lambda: Stream())

    client._stub = Stub()  # type: ignore[assignment]

    stream = client.exec_stdio_read("ta-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT)
    assert (await anext(stream)).data == b"chunk-1"
    await client.close()

    with pytest.raises(ClientClosed):
        await anext(stream)
    assert num_reads == 1
    await stream.aclose()


@pytest.mark.asyncio
async def test_stdio_stream_dropped_without_closing_releases_its_lease(make_client):
    """A caller that walks away from a stream must not pin the channel forever."""
    import gc

    client = await make_client()

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send_message(self, req, end=True):  # noqa: ARG002
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            return sr_pb2.TaskExecStdioReadResponse(data=b"chunk")

    class Stub:
        TaskExecStdioRead = UnaryStreamMethod(lambda: Stream())

    client._stub = Stub()  # type: ignore[assignment]

    stream = client.exec_stdio_read("ta-1", "exec-1", api_pb2.FILE_DESCRIPTOR_STDOUT)
    await anext(stream)
    assert client._inflight == 0

    # Dropped without aclose(); asyncio finalizes the generator, which unwinds the lease.
    del stream
    gc.collect()
    for _ in range(5):
        await asyncio.sleep(0)

    assert client._inflight == 0
    assert await released(client)


_V2_SANDBOX_ID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"


@pytest.mark.parametrize("active_lease", [False, True])
def test_synchronous_detach_runs_cleanup_on_sdk_loop(client, active_lease):
    from modal._utils.async_utils import synchronizer
    from modal.sandbox import Sandbox

    sandbox = Sandbox.from_id(_V2_SANDBOX_ID, client=client)
    cleanup_loops = []

    @synchronizer.wrap
    async def check_cleanup(raw_sandbox):
        loop = asyncio.get_running_loop()
        closed = asyncio.Event()

        class LoopCheckingChannel(FakeChannel):
            def close(self):
                cleanup_loops.append(asyncio.get_running_loop())
                super().close()
                closed.set()

        router = TaskCommandRouterClient(
            None,
            "ta-1",
            "https://router.test",
            "jwt",
            LoopCheckingChannel(),  # type: ignore[arg-type]
            loop,
            asyncio.Lock(),
        )
        router._arm_idle_release()
        if active_lease:
            router._take_lease()
        raw_sandbox._command_router_client = router
        yield
        assert router._closed
        if active_lease:
            assert not closed.is_set()
            router._drop_lease()
        await asyncio.wait_for(closed.wait(), timeout=1)
        assert cleanup_loops == [loop]
        await router.close()

    checks = check_cleanup(sandbox)
    next(checks)
    sandbox.detach()
    with pytest.raises(StopIteration):
        next(checks)


@pytest.mark.asyncio
async def test_repeated_detach_closes_channel_once(client, make_client):
    from modal._utils.async_utils import synchronizer
    from modal.sandbox import Sandbox

    sandbox = await Sandbox.from_id.aio(_V2_SANDBOX_ID, client=client)
    raw_sandbox = synchronizer._translate_in(sandbox)
    router = await make_client(idle_timeout=3600)
    raw_sandbox._command_router_client = router

    await raw_sandbox.detach()
    assert not raw_sandbox._attached
    assert router._closed
    assert router._channel.num_permanent_closes == 1

    await asyncio.wait_for(raw_sandbox.detach(), timeout=1)
    assert router._channel.num_permanent_closes == 1


@pytest.mark.asyncio
async def test_detach_suppresses_router_cleanup_errors(client):
    from modal._utils.async_utils import synchronizer
    from modal.sandbox import Sandbox

    close_calls = []

    class Router:
        async def close_when_idle(self):
            close_calls.append("router")
            raise RuntimeError("router close failed")

    sandbox = await Sandbox.from_id.aio(_V2_SANDBOX_ID, client=client)
    raw_sandbox = synchronizer._translate_in(sandbox)
    raw_sandbox._command_router_client = Router()  # type: ignore[assignment]

    await raw_sandbox.detach()
    assert close_calls == ["router"]

    await raw_sandbox.detach()
    assert close_calls == ["router"]


@pytest.mark.asyncio
async def test_router_initialized_during_detach_is_closed(client, monkeypatch):
    """A connection that finishes after detach must never be cached or returned."""
    from modal._utils.async_utils import synchronizer
    from modal.sandbox import Sandbox

    started = asyncio.Event()
    finish_connect = asyncio.Event()

    class Router:
        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    router = Router()

    async def init_v2_by_sandbox_id(*args, **kwargs):  # noqa: ARG001
        started.set()
        await finish_connect.wait()
        return router

    monkeypatch.setattr(TaskCommandRouterClient, "init_v2_by_sandbox_id", init_v2_by_sandbox_id)

    sandbox = await Sandbox.from_id.aio(_V2_SANDBOX_ID, client=client)
    raw_sandbox = synchronizer._translate_in(sandbox)
    get_router = asyncio.create_task(raw_sandbox._get_command_router_client("ta-1"))
    await started.wait()

    await raw_sandbox.detach()
    finish_connect.set()

    with pytest.raises(ClientClosed):
        await get_router
    assert router.close_calls == 1
    assert raw_sandbox._command_router_client is None


@skip_windows("Needs subprocess support")
@pytest.mark.asyncio
async def test_concurrent_first_operations_share_one_channel(servicer, client, monkeypatch):
    """Concurrent first operations share a single connection."""
    from modal._utils.async_utils import synchronizer
    from modal.sandbox import Sandbox

    num_connects = 0
    original = TaskCommandRouterClient.init_v2_by_sandbox_id

    async def _count(*args, **kwargs):
        nonlocal num_connects
        num_connects += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(TaskCommandRouterClient, "init_v2_by_sandbox_id", _count)

    sandbox = await Sandbox.from_id.aio(_V2_SANDBOX_ID, client=client)
    processes = await asyncio.gather(*(sandbox.exec.aio("echo", "hello") for _ in range(5)))
    for process in processes:
        assert await process.wait.aio() == 0

    assert num_connects == 1
    routers = {synchronizer._translate_in(sandbox)._command_router_client}
    assert len(routers) == 1


@pytest.mark.asyncio
async def test_an_arm_after_retire_is_refused():
    idle = IdleCountdown(asyncio.get_running_loop())
    idle.arm(3600, lambda: None)
    idle.retire()
    assert idle._handle is None

    idle.arm(3600, lambda: None)
    assert idle._handle is None, "an arm after retire must not start a timer"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, 3600])
async def test_expire_when_idle_accelerates_existing_and_future_countdowns(timeout):
    idle = IdleCountdown(asyncio.get_running_loop())
    released = asyncio.Event()
    idle.arm(timeout, released.set)
    handle = idle._handle
    idle.expire_when_idle()
    assert released.is_set()
    assert idle._handle is None
    if handle is not None:
        assert handle.cancelled()

    released.clear()
    idle.expire_when_idle()
    await asyncio.sleep(0.01)
    assert not released.is_set(), "repeated expiry must not invoke the released callback again"
    idle.arm(timeout, released.set)
    await asyncio.wait_for(released.wait(), timeout=1)
    idle.retire()


@pytest.mark.asyncio
async def test_expire_when_idle_does_not_release_active_work():
    idle = IdleCountdown(asyncio.get_running_loop())
    released = asyncio.Event()
    idle.arm(3600, released.set)
    idle.stop()
    idle.expire_when_idle()
    await asyncio.sleep(0.01)
    assert not released.is_set()
    idle.arm(3600, released.set)
    await asyncio.wait_for(released.wait(), timeout=1)
    idle.retire()


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_timeout", [0, 3600])
async def test_close_when_idle_waits_for_every_lease(make_client, idle_timeout):
    client = await make_client(idle_timeout=idle_timeout)
    async with client._lease():
        async with client._lease():
            await client.close_when_idle()
            await client.close_when_idle()
            with pytest.raises(ClientClosed):
                async with client._lease():
                    pass
            await asyncio.sleep(0.01)
            assert client._channel.num_permanent_closes == 0
        await asyncio.sleep(0.01)
        assert client._channel.num_permanent_closes == 0

    await asyncio.sleep(0.01)
    assert client._channel.num_permanent_closes == 1
    assert client._inflight == 0


@pytest.mark.asyncio
async def test_force_close_after_close_when_idle(make_client):
    client = await make_client()
    async with client._lease():
        await client.close_when_idle()
        await client.close()
        assert client._channel.num_permanent_closes == 1
    await client.close()
    assert client._channel.num_permanent_closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["_take_lease", "_drop_lease", "_arm_idle_release", "_release_if_idle", "_close_channel"]
)
@pytest.mark.parametrize("running_loop", [False, True])
async def test_channel_bookkeeping_requires_owning_event_loop(make_client, operation, running_loop):
    client = await make_client(idle_timeout=3600)
    handle = client._idle._handle

    async def run_on_other_loop():
        with pytest.raises(AssertionError, match="owning event loop"):
            getattr(client, operation)()

    def run_without_loop():
        with pytest.raises(RuntimeError, match="no running event loop"):
            getattr(client, operation)()

    if running_loop:
        await asyncio.to_thread(lambda: asyncio.run(run_on_other_loop()))
    else:
        await asyncio.to_thread(run_without_loop)
    assert client._inflight == 0
    assert client._idle._handle is handle
    assert handle is not None and not handle.cancelled()
    assert client._channel.num_soft_closes == 0
    assert client._channel.num_permanent_closes == 0
    await client.close()
    assert client._channel.num_permanent_closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["arm", "stop", "retire", "expire_when_idle"])
@pytest.mark.parametrize("running_loop", [False, True])
async def test_idle_bookkeeping_requires_owning_event_loop(operation, running_loop):
    idle = IdleCountdown(asyncio.get_running_loop())
    released = asyncio.Event()
    idle.arm(3600, released.set)
    handle = idle._handle
    args = (3600, released.set) if operation == "arm" else ()

    async def run_on_other_loop():
        with pytest.raises(AssertionError, match="owning event loop"):
            getattr(idle, operation)(*args)

    def run_without_loop():
        with pytest.raises(RuntimeError, match="no running event loop"):
            getattr(idle, operation)(*args)

    try:
        if running_loop:
            await asyncio.to_thread(lambda: asyncio.run(run_on_other_loop()))
        else:
            await asyncio.to_thread(run_without_loop)
        assert idle._handle is handle
        assert handle is not None and not handle.cancelled()
        assert not idle._retired
        assert not idle._expire_on_idle
        assert not released.is_set()
    finally:
        idle.retire()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["close", "close_when_idle"])
async def test_channel_lifecycle_requires_owning_event_loop(make_client, operation):
    client = await make_client(idle_timeout=3600)

    async def run_on_other_loop():
        with pytest.raises(AssertionError, match="owning event loop"):
            await getattr(client, operation)()

    await asyncio.to_thread(lambda: asyncio.run(run_on_other_loop()))
    assert not client._closed
    assert client._channel.num_permanent_closes == 0


@pytest.mark.asyncio
async def test_detach_does_not_rebuild_the_router_client(servicer, client, monkeypatch):
    """A detached Sandbox cannot open a new connection."""
    from modal._utils.async_utils import synchronizer
    from modal.sandbox import Sandbox

    connects = 0
    original = TaskCommandRouterClient.init_v2_by_sandbox_id

    async def _count(*args, **kwargs):
        nonlocal connects
        connects += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(TaskCommandRouterClient, "init_v2_by_sandbox_id", _count)

    sandbox = await Sandbox.from_id.aio(_V2_SANDBOX_ID, client=client)
    await sandbox.detach.aio()

    inner = synchronizer._translate_in(sandbox)
    with pytest.raises(ClientClosed):
        await inner._get_command_router_client("ta-1")
    assert connects == 0, "a detached Sandbox reconnected"
