# Copyright Modal Labs 2023
# type: ignore
import os
import secrets
import socket
import subprocess
import threading
import time
import webbrowser
from typing import Any, Dict

from modal import Image, Queue, Stub, forward

args: Dict[str, Any] = {}

stub = Stub()
stub.image = Image.from_registry(args.get("image"), add_python=args.get("add_python")).pip_install("jupyterlab")


def wait_for_port(url: str, q: Queue):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 8888), timeout=15.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 15.0:
                raise TimeoutError("Waited too long for port 8888 to accept connections") from exc
    q.put(url)


@stub.function(cpu=args.get("cpu"), memory=args.get("memory"), gpu=args.get("gpu"), timeout=args.get("timeout"))
def run_jupyter(q: Queue):
    os.mkdir("/lab")
    token = secrets.token_urlsafe(13)
    with forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        threading.Thread(target=wait_for_port, args=(url, q)).start()
        subprocess.run(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                "--port=8888",
                "--notebook-dir=/lab",
                "--LabApp.allow_origin='*'",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )
    q.put("done")


@stub.local_entrypoint()
def main():
    with Queue.ephemeral() as q:
        stub.run_jupyter.spawn(q)
        url = q.get()
        time.sleep(1)  # Give Jupyter a chance to start up
        print("\nJupyter on Modal, opening in browser...")
        print(f"   -> {url}\n")
        webbrowser.open(url)
        assert q.get() == "done"
