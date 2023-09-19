# Copyright Modal Labs 2023
# type: ignore
import os
import secrets
import subprocess
import time
import webbrowser

from modal import Image, Queue, Stub
from modal._relay_client import forward

stub = Stub()
stub.image = Image.debian_slim().pip_install("jupyterlab")
stub.q = Queue.new()

token = secrets.token_urlsafe(13)


@stub.function(timeout=3600)
def run_jupyter():
    with forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        stub.q.put(url)
        subprocess.run(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--port=8888",
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
