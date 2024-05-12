# Copyright Modal Labs 2023
# type: ignore
import json
import os
import secrets
import socket
import subprocess
import threading
import time
import webbrowser
from typing import Any, Dict

from modal import App, Image, Mount, Queue, Secret, Volume, forward

# Passed by `modal launch` locally via CLI, plumbed to remote runner through secrets.
args: Dict[str, Any] = json.loads(os.environ.get("MODAL_LAUNCH_ARGS", "{}"))


app = App()
app.image = Image.from_registry(args.get("image"), add_python=args.get("add_python")).pip_install("jupyterlab")


mount = (
    Mount.from_local_dir(
        args.get("mount"),
        remote_path="/root/lab/mount",
    )
    if args.get("mount")
    else None
)
mounts = [mount] if mount else []

volume = (
    Volume.from_name(
        args.get("volume"),
        create_if_missing=True,
    )
    if args.get("volume")
    else None
)
volumes = {"/root/lab/volume": volume} if volume else {}


def wait_for_port(url: str, q: Queue):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 8888), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError("Waited too long for port 8888 to accept connections") from exc
    q.put(url)


@app.function(
    cpu=args.get("cpu"),
    memory=args.get("memory"),
    gpu=args.get("gpu"),
    timeout=args.get("timeout"),
    secrets=[Secret.from_dict({"MODAL_LAUNCH_ARGS": json.dumps(args)})],
    mounts=mounts,
    volumes=volumes,
    concurrency_limit=1 if volume else None,
    _allow_background_volume_commits=True if volume else False,
)
def run_jupyter(q: Queue):
    os.makedirs("/root/lab", exist_ok=True)
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
                "--notebook-dir=/root/lab",
                "--LabApp.allow_origin='*'",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )
    q.put("done")


@app.local_entrypoint()
def main():
    with Queue.ephemeral() as q:
        run_jupyter.spawn(q)
        url = q.get()
        time.sleep(1)  # Give Jupyter a chance to start up
        print("\nJupyter on Modal, opening in browser...")
        print(f"   -> {url}\n")
        webbrowser.open(url)
        assert q.get() == "done"
