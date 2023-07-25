# Copyright Modal Labs 2022
import pytest
import queue

from modal import Queue, Stub


def test_queue(servicer, client):
    stub = Stub()
    stub.q = Queue.new()
    with stub.run(client=client) as app:
        app.q.put(42)
        assert app.q.get() == 42
        with pytest.raises(queue.Empty):
            app.q.get(timeout=0)


def test_queue_use_provider(servicer, client):
    stub = Stub()
    stub.q = Queue.new()
    with stub.run(client=client):
        assert isinstance(stub.q, Queue)
        stub.q.put("xyz")
