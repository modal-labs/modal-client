import pytest
import threading
import time

import modal
from modal.client import Client
from modal_proto import api_pb2


def close_client_soon(client):
    def cb():
        time.sleep(0.1)
        print("Closing client")
        client._close()
        print("Closed")

    threading.Thread(target=cb).start()


def test_shutdown_deadlock(servicer):
    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        with modal.Queue.ephemeral(client=client) as q:
            close_client_soon(client)
            with pytest.raises(modal.client.ClientShutdown):
                q.get()
