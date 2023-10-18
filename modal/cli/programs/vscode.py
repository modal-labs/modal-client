# Copyright Modal Labs 2023
# type: ignore
import os
import secrets
import socket
import subprocess
import threading
import time
import webbrowser
from typing import Any

from modal import Image, Queue, Stub, forward

args: Any = {}

stub = Stub()
stub.image = Image.from_registry("codercom/code-server", add_python="3.11").dockerfile_commands("ENTRYPOINT []")
stub.q = Queue.new()


def wait_for_port(data):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 8080), timeout=5.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 5.0:
                raise TimeoutError("Waited too long for port 8080 to accept connections") from exc
    stub.q.put(data)


@stub.function(cpu=args["cpu"], memory=args["memory"], gpu=args["gpu"], timeout=args["timeout"])
def run_jupyter():
    os.chdir("/home/coder")
    token = secrets.token_urlsafe(13)
    with forward(8080) as tunnel:
        url = tunnel.url
        threading.Thread(target=wait_for_port, args=((url, token),)).start()
        subprocess.run(
            ["/usr/bin/entrypoint.sh", "--bind-addr", "0.0.0.0:8080", "."],
            env={**os.environ, "SHELL": "/bin/bash", "PASSWORD": token},
        )
    stub.q.put("done")


@stub.local_entrypoint()
def main():
    stub.run_jupyter.spawn()
    url, token = stub.q.get()
    time.sleep(1)  # Give Jupyter a chance to start up
    print("\nVS Code on Modal, opening in browser...")
    print(f"   -> {url}")
    print(f"   -> password: {token}\n")
    webbrowser.open(url)
    assert stub.q.get() == "done"
