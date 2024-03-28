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

from modal import Image, Queue, Stub, forward

# Passed by `modal launch` locally via CLI, empty on remote runner.
args: Dict[str, Any] = json.loads(os.environ.get("MODAL_LAUNCH_LOCAL_ARGS", "{}"))


stub = Stub()
stub.image = Image.from_registry("codercom/code-server", add_python="3.11").dockerfile_commands("ENTRYPOINT []")


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


@stub.function(cpu=args.get("cpu"), memory=args.get("memory"), gpu=args.get("gpu"), timeout=args.get("timeout"))
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


@stub.local_entrypoint()
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
