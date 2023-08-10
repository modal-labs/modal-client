# Copyright Modal Labs 2022
import pytest
import queue

from modal import Queue, Stub


def test_queue(servicer, client):
    stub = Stub()
    stub.q = Queue.new()
    with stub.run(client=client):
        assert isinstance(stub.q, Queue)
        stub.q.put(42)
        assert stub.q.get() == 42
        with pytest.raises(queue.Empty):
            stub.q.get(timeout=0)
