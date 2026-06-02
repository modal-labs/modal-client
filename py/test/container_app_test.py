# Copyright Modal Labs 2022
import asyncio
import json
import os
import pytest
import threading
from contextlib import contextmanager
from unittest import mock

from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import Message

import modal._runtime.container_io_manager as container_io_manager
from modal import App, interact
from modal._runtime.container_io_manager import ContainerIOManager, _ContainerIOManager
from modal._utils.async_utils import synchronize_api, synchronizer
from modal.exception import InvalidError
from modal.running_app import RunningApp
from modal_proto import api_pb2


def my_f_1(x):
    pass


def temp_restore_path(tmpdir):
    # Write out a restore file so that snapshot+restore will complete
    restore_path = tmpdir.join("fake-restore-state.json")
    restore_path.write_text(
        json.dumps(
            {
                "task_id": "ta-i-am-restored",
                "task_secret": "ts-i-am-restored",
                "function_id": "fu-i-am-restored",
            }
        ),
        encoding="utf-8",
    )
    return restore_path


def _container_heartbeat_requests(servicer):
    return [req for req in servicer.requests if isinstance(req, api_pb2.ContainerHeartbeatRequest)]


def _container_checkpoint_requests(servicer):
    return [req for req in servicer.requests if isinstance(req, api_pb2.ContainerCheckpointRequest)]


async def _wait_for_container_heartbeat(servicer, min_count: int = 1, timeout: float = 0.5):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if len(_container_heartbeat_requests(servicer)) >= min_count:
            return
        await asyncio.sleep(0.01)
    assert len(_container_heartbeat_requests(servicer)) >= min_count


@pytest.mark.asyncio
async def test_container_function_lazily_imported(container_client):
    function_ids: dict[str, str] = {
        "my_f_1": "fu-123",
    }
    object_handle_metadata: dict[str, Message | None] = {
        "fu-123": api_pb2.FunctionHandleMetadata(),
    }
    container_app = RunningApp("ap-123", function_ids=function_ids, object_handle_metadata=object_handle_metadata)
    app = App()

    # This is normally done in _container_entrypoint
    app._init_container(container_client, container_app)

    # Now, let's create my_f after the app started running and make sure it works
    my_f_container = app.function()(my_f_1)
    assert await my_f_container.remote.aio(42) == 1764  # type: ignore


def square(x):
    pass


@synchronize_api
async def stop_app(client, app_id):
    # helper to ensur we run the rpc from the synchronicity loop - otherwise we can run into weird deadlocks
    return await client.stub.AppStop(api_pb2.AppStopRequest(app_id=app_id))


@contextmanager
def set_env_vars(restore_path, container_addr):
    with mock.patch.dict(
        os.environ,
        {
            "MODAL_RESTORE_STATE_PATH": str(restore_path),
            "MODAL_SERVER_URL": container_addr,
            "MODAL_TASK_ID": "ta-123",
            "MODAL_IS_REMOTE": "1",
        },
    ):
        yield


@pytest.mark.asyncio
async def test_container_snapshot_reference_capture(container_client, tmpdir, servicer, client):
    app = App(include_source=False)
    from modal import Function
    from modal.runner import deploy_app

    app.function()(square)
    app_name = "my-app"
    app_id = (await deploy_app.aio(app, app_name, client=container_client)).app_id
    f = Function.from_name(app_name, "square", client=container_client)
    await f.remote.aio()
    assert f.object_id == "fu-1"
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(checkpoint_id="ch-123"), container_client)
    restore_path = temp_restore_path(tmpdir)
    with set_env_vars(restore_path, servicer.container_addr):
        await io_manager.get_task_lifecycle_manager().memory_snapshot.aio()

    # Stop the App, invalidating the fu- ID stored in `f`.
    await stop_app.aio(client, app_id)
    # After snapshot-restore the previously looked-up Function should get refreshed and have the
    # new fu- ID. ie. the ID should not be stale and invalid.
    new_app_id = (await deploy_app.aio(app, app_name, client=client)).app_id
    assert new_app_id != app_id
    await f.remote.aio()
    assert f.object_id == "fu-2"
    # Purposefully break FunctionGet to check the hydration is cached.
    del servicer.app_objects[new_app_id]
    await f.remote.aio()  # remote call succeeds because it didn't re-hydrate Function
    assert f.object_id == "fu-2"


def test_container_snapshot_restore_heartbeats(tmpdir, servicer, container_client, monkeypatch):
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(checkpoint_id="ch-123"), container_client)
    restore_path = temp_restore_path(tmpdir)
    heartbeat_interval_secs = 0.01
    fake_time = 0.0
    monotonic_calls = 0
    heartbeat_loop_started = threading.Event()
    heartbeat_sent = threading.Event()

    def fake_monotonic():
        nonlocal fake_time, monotonic_calls
        monotonic_calls += 1
        fake_time += heartbeat_interval_secs
        if monotonic_calls >= 2:
            heartbeat_loop_started.set()
        return fake_time

    async def container_heartbeat(_servicer, stream):
        await stream.recv_message()
        heartbeat_sent.set()
        await stream.send_message(api_pb2.ContainerHeartbeatResponse())

    monkeypatch.setattr(container_io_manager, "HEARTBEAT_INTERVAL", heartbeat_interval_secs)
    monkeypatch.setattr(container_io_manager, "time", mock.Mock(monotonic=fake_monotonic))

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerHeartbeat", container_heartbeat)
        # Ensure that heartbeats only run after the snapshot.
        with io_manager.heartbeats(True):
            assert heartbeat_loop_started.wait(timeout=1.0)
            assert not ctx.get_requests("ContainerHeartbeat")
            with set_env_vars(restore_path, servicer.container_addr):
                with io_manager.snapshot_context_manager():
                    io_manager.get_task_lifecycle_manager().memory_snapshot()
            assert heartbeat_sent.wait(timeout=1.0)
            assert ctx.get_requests("ContainerHeartbeat")


@pytest.mark.asyncio
async def test_container_snapshot_restore_pauses_heartbeats_during_memory_snapshot(
    tmpdir, servicer, container_client, monkeypatch
):
    io_manager = _ContainerIOManager(
        api_pb2.ContainerArguments(checkpoint_id="ch-123"),
        synchronizer._translate_in(container_client),
    )
    heartbeat_interval_secs = 0.01
    checkpoint_started = asyncio.Event()
    finish_checkpoint = asyncio.Event()
    restore_path = temp_restore_path(tmpdir)

    async def container_checkpoint(request):
        checkpoint_started.set()
        await finish_checkpoint.wait()
        return Empty()

    monkeypatch.setattr(io_manager._client.stub, "ContainerCheckpoint", container_checkpoint)
    monkeypatch.setattr("modal._runtime.container_io_manager.HEARTBEAT_INTERVAL", heartbeat_interval_secs)
    servicer.container_heartbeat_abort.set()

    async def run_memory_snapshot():
        with set_env_vars(restore_path, servicer.container_addr):
            async with io_manager.snapshot_context_manager():
                await io_manager._task_lifecycle_manager.memory_snapshot()

    async with io_manager.heartbeats(True):
        await asyncio.sleep(heartbeat_interval_secs * 2)
        assert not _container_heartbeat_requests(servicer)

        snapshot_task = asyncio.create_task(run_memory_snapshot())
        try:
            await asyncio.wait_for(checkpoint_started.wait(), timeout=1.0)
            heartbeat_count = len(_container_heartbeat_requests(servicer))

            await asyncio.sleep(heartbeat_interval_secs * 3)
            assert len(_container_heartbeat_requests(servicer)) == heartbeat_count

            finish_checkpoint.set()
            await asyncio.wait_for(snapshot_task, timeout=1.0)
            await _wait_for_container_heartbeat(servicer, min_count=heartbeat_count + 1)
        finally:
            finish_checkpoint.set()
            if not snapshot_task.done():
                snapshot_task.cancel()
                await asyncio.gather(snapshot_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_container_gpu_snapshot_failure_unpauses_heartbeats(servicer, container_client, monkeypatch):
    function_def = api_pb2.Function(
        _experimental_enable_gpu_snapshot=True,
        resources=api_pb2.Resources(gpu_config=api_pb2.GPUConfig(gpu_type="A100", count=1)),
    )
    io_manager = _ContainerIOManager(
        api_pb2.ContainerArguments(checkpoint_id="ch-123", function_def=function_def),
        synchronizer._translate_in(container_client),
    )
    heartbeat_interval_secs = 0.01
    checkpoint_session = mock.Mock()
    checkpoint_session.checkpoint.side_effect = RuntimeError("gpu checkpoint failed")
    monkeypatch.setattr(
        "modal._runtime.task_lifecycle_manager.gpu_memory_snapshot.CudaCheckpointSession",
        mock.Mock(return_value=checkpoint_session),
    )
    monkeypatch.setattr("modal._runtime.container_io_manager.HEARTBEAT_INTERVAL", heartbeat_interval_secs)
    servicer.container_heartbeat_abort.set()

    async with io_manager.heartbeats(True):
        await asyncio.sleep(heartbeat_interval_secs * 2)
        assert not _container_heartbeat_requests(servicer)

        with pytest.raises(RuntimeError, match="gpu checkpoint failed"):
            async with io_manager.snapshot_context_manager():
                await io_manager.get_task_lifecycle_manager().memory_snapshot.aio()

        assert not io_manager._waiting_for_memory_snapshot
        assert not _container_checkpoint_requests(servicer)
        await _wait_for_container_heartbeat(servicer)


@pytest.mark.asyncio
async def test_container_checkpoint_failure_unpauses_heartbeats(servicer, container_client, monkeypatch):
    io_manager = _ContainerIOManager(
        api_pb2.ContainerArguments(checkpoint_id="ch-123"),
        synchronizer._translate_in(container_client),
    )
    heartbeat_interval_secs = 0.01

    async def raise_checkpoint_error(request):
        raise RuntimeError("container checkpoint failed")

    monkeypatch.setattr(io_manager._client.stub, "ContainerCheckpoint", raise_checkpoint_error)
    monkeypatch.setattr("modal._runtime.container_io_manager.HEARTBEAT_INTERVAL", heartbeat_interval_secs)
    servicer.container_heartbeat_abort.set()

    async with io_manager.heartbeats(True):
        await asyncio.sleep(heartbeat_interval_secs * 2)
        assert not _container_heartbeat_requests(servicer)

        with pytest.raises(RuntimeError, match="container checkpoint failed"):
            async with io_manager.snapshot_context_manager():
                await io_manager.get_task_lifecycle_manager().memory_snapshot.aio()

        assert not io_manager._waiting_for_memory_snapshot
        await _wait_for_container_heartbeat(servicer)


@pytest.mark.asyncio
async def test_container_debug_snapshot(container_client, tmpdir, servicer):
    # Get an IO manager, where restore takes place
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(checkpoint_id="ch-123"), container_client)
    restore_path = tmpdir.join("fake-restore-state.json")
    # Write the restore file to start a debugger
    restore_path.write_text(
        json.dumps({"snapshot_debug": "1"}),
        encoding="utf-8",
    )

    # Test that the breakpoint was called
    test_breakpoint = mock.Mock()
    with mock.patch("sys.breakpointhook", test_breakpoint):
        with set_env_vars(restore_path, servicer.container_addr):
            await io_manager.get_task_lifecycle_manager().memory_snapshot.aio()
            test_breakpoint.assert_called_once()


@pytest.mark.asyncio
async def test_rpc_wrapping_restores(container_client, servicer, tmpdir, client):
    import modal

    io_manager = ContainerIOManager(api_pb2.ContainerArguments(checkpoint_id="ch-123"), container_client)
    restore_path = temp_restore_path(tmpdir)

    d = modal.Dict.from_name("my-amazing-dict", create_if_missing=True, client=client)
    await d.put.aio("xyz", 123)
    await d.put.aio("abc", 123)

    with set_env_vars(restore_path, servicer.container_addr):
        await io_manager.get_task_lifecycle_manager().memory_snapshot.aio()

    # TODO(Jonathon): These RPC wrappers are tested directly because I could not
    # find a way to construct in this test a UnaryStreamWrapper with a stale snapshotted client.
    @synchronize_api
    async def exercise_rpcs():
        n = 0
        # Test UnaryStreamWrapper
        async for _ in container_client.stub.DictContents.unary_stream(
            api_pb2.DictContentsRequest(dict_id=d.object_id, keys=True)
        ):
            n += 1
        assert n == 2
        # Test UnaryUnaryWrapper
        await container_client.stub.DictClear(api_pb2.DictClearRequest(dict_id=d.object_id))

    await exercise_rpcs.aio()


def test_interact(container_client, servicer):
    # Initialize container singleton
    function_def = api_pb2.Function(pty_info=api_pb2.PTYInfo(pty_type=api_pb2.PTYInfo.PTY_TYPE_SHELL))
    ContainerIOManager(api_pb2.ContainerArguments(function_def=function_def), container_client)
    with servicer.intercept() as ctx:
        ctx.add_response("FunctionStartPtyShell", Empty())
        interact()


def test_interact_no_pty_error(container_client, servicer):
    # Initialize container singleton
    ContainerIOManager(api_pb2.ContainerArguments(), container_client)
    with pytest.raises(InvalidError, match=r"modal.interact\(\) without running Modal in interactive mode"):
        interact()
