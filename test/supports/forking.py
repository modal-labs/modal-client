# Copyright Modal Labs 2025
import os

from modal._utils.async_utils import synchronize_api
from modal.client import Client
from modal_proto import api_pb2


@synchronize_api
async def list_volumes(method):
    await method(api_pb2.VolumeListRequest(environment_name="main"))
    print(os.getpid())


def sub(api_stub):
    list_volumes(api_stub)


if __name__ == "__main__":
    client = Client.from_env()
    rpc_method = client.stub.VolumeList
    sub(rpc_method)
    if not (fork_pid := os.fork()):
        list_volumes(rpc_method)
    else:
        os.waitpid(fork_pid, 0)
