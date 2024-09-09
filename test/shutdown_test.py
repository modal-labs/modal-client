# Copyright Modal Labs 2024
import pytest
import threading
import time

import modal
from modal.client import Client
from modal_proto import api_pb2


def close_client_soon(client):
    def cb():
        time.sleep(0.1)
        client._close()

    threading.Thread(target=cb).start()


@pytest.mark.timeout(5)
def test_shutdown_deadlock(servicer):
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
