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

from modal import Image, Queue, Stub
from modal._relay_client import forward

args: Any = {}

stub = Stub()
stub.image = Image.from_registry(args["image"], add_python=args["add_python"]).pip_install("jupyterlab")
stub.q = Queue.new()


def wait_for_port(url: str):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 8888), timeout=5.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 5.0:
                raise TimeoutError("Waited too long for port 8888 to accept connections") from exc
    stub.q.put(url)


@stub.function(cpu=args["cpu"], memory=args["memory"], gpu=args["gpu"], timeout=args["timeout"])
def run_jupyter():
    os.mkdir("/lab")
    token = secrets.token_urlsafe(13)
    with forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        threading.Thread(target=wait_for_port, args=(url,)).start()
        subprocess.run(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--port=8888",
                "--notebook-dir=/lab",
                "--LabApp.allow_origin='*'",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )
    stub.q.put("done")


@stub.local_entrypoint()
def main():
    stub.run_jupyter.spawn()
    url = stub.q.get()
    time.sleep(1)  # Give Jupyter a chance to start up
    print("\nJupyter on Modal, opening in browser...")
    print(f"   -> {url}\n")
    webbrowser.open(url)
    assert stub.q.get() == "done"
