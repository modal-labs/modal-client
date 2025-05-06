# Copyright Modal Labs 2024
import os
import sys

from modal._utils.async_utils import synchronize_api
from modal.client import Client
from modal.config import config
from modal_proto import api_pb2


@synchronize_api
async def test_stub(stub):
    await stub.VolumeList(api_pb2.VolumeListRequest(environment_name="main"))
    print(os.getpid())


@synchronize_api
async def test_stub_method(volume_list_stub_method):
    await volume_list_stub_method(api_pb2.VolumeListRequest(environment_name="main"))
    print(os.getpid())


if __name__ == "__main__":
    client = Client.from_env()
    test = sys.argv[1]

    new_proc = False
    if test == "test_stub_method":
        # Test that a reference to a stub method can be used across forks
        rpc_method = client.stub.VolumeList
        test_stub_method(rpc_method)
        if not (fork_pid := os.fork()):
            new_proc = True
            test_stub_method(rpc_method)
        else:
            os.waitpid(fork_pid, 0)
    elif test == "test_stub_reference":
        # Test that a reference to a stub can be used across forks
        stub = client.get_stub(config["server_url"])
        test_stub(stub)
        if not (fork_pid := os.fork()):
            test_stub(stub)
        else:
            os.waitpid(fork_pid, 0)
    elif test == "test_default_stub":
        # Test that the default stub can be used across forks
        test_stub(client.stub)
        if not (fork_pid := os.fork()):
            test_stub(client.stub)
        else:
            os.waitpid(fork_pid, 0)
    elif test == "test_default_stub_reference":
        # Test that a reference to the  default stub can be used across forks
        stub = client.stub
        test_stub(stub)
        if not (fork_pid := os.fork()):
            test_stub(stub)
        else:
            os.waitpid(fork_pid, 0)
    else:
        raise ValueError(f"Unknown test: {test}")
