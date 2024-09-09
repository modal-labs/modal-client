# Copyright Modal Labs 2024
import threading

from modal.client import ClientClosed
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
    except ClientClosed:
        print("Graceful shutdown")
