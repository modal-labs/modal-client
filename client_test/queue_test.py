# Copyright Modal Labs 2022
import queue
import pytest

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
    with stub.run(client=client) as app:
        stub.q.put("xyz")
