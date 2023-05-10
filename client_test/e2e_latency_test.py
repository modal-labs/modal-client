import asyncio
import os
import time
import unittest.mock
from pathlib import Path

import pytest
from grpclib.stream import send_message

import modal
from modal import Client
from modal_proto import api_pb2


@pytest.fixture()
def grpc_requests(servicer):
    recorded_requests = []

    async def slow_send(stream, codec, message, message_type, **kwargs):
        await asyncio.sleep(0.2)
        recorded_requests.append(message_type)
        return await send_message(stream, codec, message, message_type, **kwargs)

    with unittest.mock.patch("grpclib.client.send_message", slow_send):
        yield recorded_requests


stub = modal.Stub()


@stub.function()
def func():
    pass


@pytest.mark.asyncio
@unittest.mock.patch("modal._function_utils.FunctionInfo._get_auto_mounts", lambda x: {})
async def test_client(servicer, grpc_requests, test_dir):
    from modal_test_support import hello

    os.chdir(Path(hello.__file__).parent)
    t0 = time.monotonic()
    with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        with hello.stub.run(client=client):
            hello.hello.call()

    for req in grpc_requests:
        print(req)
    dur = time.monotonic() - t0
    print(dur)


# 0, 0.5
# 0.05, 2.33
# 0.1, 3.85
# 0.2, 6.77
# 0.3, 10.28
