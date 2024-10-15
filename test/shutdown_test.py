# Copyright Modal Labs 2024
import asyncio
import pytest
import threading
import time

import grpclib

import modal
from modal._utils.async_utils import synchronize_api
from modal.client import Client
from modal.exception import ClientClosed
from modal_proto import api_pb2


def close_client_soon(client):
    def cb():
        time.sleep(0.1)
        client._close()

    threading.Thread(target=cb).start()


@pytest.mark.timeout(5)
def test_client_shutdown_raises_client_closed(servicer):
    # Queue.get() loops rpc calls until it gets a response - make sure it shuts down
    # if the client is closed and doesn't stay in an indefinite retry loop
    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        with modal.Queue.ephemeral(client=client) as q:
            close_client_soon(client)  # simulate an early shutdown of the client
            with pytest.raises(modal.exception.ClientClosed):
                # ensure that ongoing rcp calls are aborted
                q.get()

            with pytest.raises(modal.exception.ClientClosed):
                # ensure the client isn't doesn't allow for *new* connections
                # after shutdown either
                q.get()


@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_client_shutdown_raises_client_closed_streaming(servicer, caplog):
    # Queue.get() loops rpc calls until it gets a response - make sure it shuts down
    # if the client is closed and doesn't stay in an indefinite retry loop

    async def _mocked_logs_loop(client: Client, app_id: str):
        request = api_pb2.AppGetLogsRequest(
            app_id=app_id,
            task_id="",
            timeout=55,
            last_entry_id="",
        )
        async for _ in client.stub.AppGetLogs.unary_stream(request):
            pass

    sync_log_loop = synchronize_api(_mocked_logs_loop)

    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        t = asyncio.create_task(sync_log_loop.aio(client, "ap-1"))
        await asyncio.sleep(0.1)  # in loop

    with pytest.raises(ClientClosed):
        await t

    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        t = asyncio.create_task(_mocked_logs_loop(client, "ap-1"))
        await asyncio.sleep(0.1)  # in loop

    with pytest.raises(grpclib.exceptions.StreamTerminatedError):
        await t
    assert len(caplog.records) == 3  # open, send and recv called outside of task context
    for rec in caplog.records:
        assert "made outside of task context" in rec.message


@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_client_close_cancellation_context_only_used_in_correct_event_loop(servicer, caplog):
    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        with modal.Queue.ephemeral(client=client) as q:
            request = api_pb2.QueueGetRequest(
                queue_id=q.object_id,
                partition_key=b"",
                timeout=10,
                n_values=1,
            )
            # this request should not use task context since it's not issued from the same loop
            # that the task context is triggered from, otherwise we'll get cross-event loop
            # waits/cancellations etc.
            t = asyncio.create_task(client.stub.QueueGet(request))
            await asyncio.sleep(0.1)
    with pytest.raises(grpclib.exceptions.StreamTerminatedError):
        await t
    assert len(caplog.records) == 1
    assert "QueueGet made outside of task context" in caplog.records[0].message
