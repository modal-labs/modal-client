import threading

from modal._utils.grpc_utils import ClientShutdown
from modal.queue import Queue

event = threading.Event()


def stop_soon():
    event.wait()
    print("stopping rpcs")
    print("bye")


t = threading.Thread(target=stop_soon)
t.start()


with Queue.ephemeral() as q:
    try:
        event.set()
        q.get()
    except ClientShutdown:
        print("Graceful shutdown")
