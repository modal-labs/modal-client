# Copyright Modal Labs 2025
# type: ignore
import json
import os
import secrets
import socket
import subprocess
import threading
import time
import webbrowser
from typing import Any

from modal import App, Image, Queue, Secret, Volume, forward

# Args injected by `modal launch` CLI.
args: dict[str, Any] = json.loads(os.environ.get("MODAL_LAUNCH_ARGS", "{}"))

app = App()

image = Image.from_registry(args.get("image"), add_python=args.get("add_python")).uv_pip_install("marimo")

# Optional host-filesystem mount (read-only snapshot of your project, useful for editing)
if args.get("mount"):
    image = image.add_local_dir(args["mount"], remote_path="/root/marimo/mount")

# Optional persistent Modal volume
volume = Volume.from_name(args["volume"], create_if_missing=True) if args.get("volume") else None
volumes = {"/root/marimo/volume": volume} if volume else {}


def _wait_for_port(url: str, q: Queue) -> None:
    start = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 8888), timeout=30):
                break
        except OSError as exc:
            if time.monotonic() - start > 30:
                raise TimeoutError("marimo server did not start within 30 s") from exc
            time.sleep(0.05)
    q.put(url)


@app.function(
    image=image,
    cpu=args.get("cpu"),
    memory=args.get("memory"),
    gpu=args.get("gpu"),
    timeout=args.get("timeout"),
    secrets=[Secret.from_dict({"MODAL_LAUNCH_ARGS": json.dumps(args)})],
    volumes=volumes,
    max_containers=1 if volume else None,
)
def run_marimo(q: Queue):
    os.makedirs("/root/marimo", exist_ok=True)

    # marimo supports token-based auth; generate one so only you can connect
    token = secrets.token_urlsafe(12)

    with forward(8888) as tunnel:
        url = f"{tunnel.url}/?access_token={token}"
        threading.Thread(target=_wait_for_port, args=(url, q), daemon=True).start()

        print("\nmarimo on Modal, opening in browser …")
        print(f"   -> {url}\n")

        # Launch the headless edit server
        subprocess.run(
            [
                "marimo",
                "edit",
                "--headless",  # don't open browser in container
                "--host",
                "0.0.0.0",  # bind all interfaces
                "--port",
                "8888",
                "--token-password",
                token,  # enable session-based auth
                "--skip-update-check",
                "/root/marimo",  # workspace directory
            ],
            env={**os.environ, "SHELL": "/bin/bash"},
        )

    q.put("done")


@app.local_entrypoint()
def main():
    with Queue.ephemeral() as q:
        run_marimo.spawn(q)
        url = q.get()  # first message = connect URL
        time.sleep(1)  # give server a heartbeat
        webbrowser.open(url)
        assert q.get() == "done"
