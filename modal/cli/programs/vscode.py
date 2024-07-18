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
from typing import Any, Dict, Tuple

from modal import App, Image, Mount, Queue, Secret, Volume, forward

# Passed by `modal launch` locally via CLI, plumbed to remote runner through secrets.
args: Dict[str, Any] = json.loads(os.environ.get("MODAL_LAUNCH_ARGS", "{}"))


app = App()
app.image = Image.from_registry("codercom/code-server", add_python="3.11").dockerfile_commands("ENTRYPOINT []")


mount = (
    Mount.from_local_dir(
        args.get("mount"),
        remote_path="/home/coder/mount",
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
volumes = {"/home/coder/volume": volume} if volume else {}


def wait_for_port(data: Tuple[str, str], q: Queue):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 8080), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError("Waited too long for port 8080 to accept connections") from exc
    q.put(data)


@app.function(
    cpu=args.get("cpu"),
    memory=args.get("memory"),
    gpu=args.get("gpu"),
    timeout=args.get("timeout"),
    secrets=[Secret.from_dict({"MODAL_LAUNCH_ARGS": json.dumps(args)})],
    mounts=mounts,
    volumes=volumes,
    concurrency_limit=1 if volume else None,
)
def run_vscode(q: Queue):
    os.chdir("/home/coder")
    token = secrets.token_urlsafe(13)
    with forward(8080) as tunnel:
        url = tunnel.url
        threading.Thread(target=wait_for_port, args=((url, token), q)).start()
        subprocess.run(
            ["/usr/bin/entrypoint.sh", "--bind-addr", "0.0.0.0:8080", "."],
            env={**os.environ, "SHELL": "/bin/bash", "PASSWORD": token},
        )
    q.put("done")


@app.local_entrypoint()
def main():
    with Queue.ephemeral() as q:
        run_vscode.spawn(q)
        url, token = q.get()
        time.sleep(1)  # Give VS Code a chance to start up
        print("\nVS Code on Modal, opening in browser...")
        print(f"   -> {url}")
        print(f"   -> password: {token}\n")
        webbrowser.open(url)
        assert q.get() == "done"
