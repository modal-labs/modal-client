# Copyright Modal Labs 2022
import pytest
import queue
import time

from modal import Queue, Stub
from modal.exception import DeprecationError

from .supports.skip import skip_macos, skip_windows


def test_queue(servicer, client):
    stub = Stub()
    with pytest.warns(DeprecationError):
        stub.q = Queue.new()
    with stub.run(client=client):
        assert isinstance(stub.q, Queue)
        assert stub.q.len() == 0
        stub.q.put(42)
        assert stub.q.len() == 1
        assert stub.q.get() == 42
        with pytest.raises(queue.Empty):
            stub.q.get(timeout=0)
        assert stub.q.len() == 0


def test_queue_ephemeral(servicer, client):
    with Queue.ephemeral(client=client, _heartbeat_sleep=1) as q:
        q.put("hello")
        assert q.len() == 1
        assert q.get() == "hello"
        time.sleep(1.5)  # enough to trigger two heartbeats

    assert servicer.n_queue_heartbeats == 2


@skip_macos("TODO(erikbern): this consistently fails on OSX. Unclear why.")
@skip_windows("TODO(Jonathon): figure out why timeouts don't occur on Windows.")
@pytest.mark.parametrize(
    ["put_timeout_secs", "min_queue_full_exc_count", "max_queue_full_exc_count"],
    [
        (0.02, 1, 100),  # a low timeout causes some exceptions
        (10.0, 0, 0),  # a high timeout causes zero exceptions
        (0.00, 1, 100),  # zero-len timeout causes some exceptions
        (None, 0, 0),  # no timeout causes zero exceptions
    ],
)
def test_queue_blocking_put(put_timeout_secs, min_queue_full_exc_count, max_queue_full_exc_count, servicer, client):
    import queue
    import threading

    producer_delay = 0.001
    consumer_delay = producer_delay * 5

    queue_full_exceptions = 0
    with Queue.ephemeral(client=client) as q:

        def producer():
            nonlocal queue_full_exceptions
            for i in range(servicer.queue_max_len * 2):
                item = f"Item {i}"
                try:
                    q.put(item, block=True, timeout=put_timeout_secs)  # type: ignore
                except queue.Full:
                    queue_full_exceptions += 1
                time.sleep(producer_delay)

        def consumer():
            while True:
                time.sleep(consumer_delay)
                item = q.get(block=True)  # type: ignore
                if item is None:
                    break  # Exit if a None item is received

        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        producer_thread.start()
        consumer_thread.start()
        producer_thread.join()
        # Stop the consumer by sending a None item
        q.put(None)  # type: ignore
        consumer_thread.join()

        assert queue_full_exceptions >= min_queue_full_exc_count
        assert queue_full_exceptions <= max_queue_full_exc_count


def test_queue_nonblocking_put(servicer, client):
    with Queue.ephemeral(client=client) as q:
        # Non-blocking PUTs don't tolerate a full queue and will raise exception.
        with pytest.raises(queue.Full) as excinfo:
            for i in range(servicer.queue_max_len + 1):
                q.put(i, block=False)  # type: ignore

    assert str(servicer.queue_max_len) in str(excinfo.value)
    assert i == servicer.queue_max_len


def test_queue_deploy(servicer, client):
    d = Queue.lookup("xyz", create_if_missing=True, client=client)
    d.put(123)
