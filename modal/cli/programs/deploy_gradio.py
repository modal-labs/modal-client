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
from typing import Any
import argparse

import modal
from modal import App, Image, Queue, Secret, Volume, forward
from modal.cli.import_refs import ImportRef

# Passed by `modal launch` locally via CLI, plumbed to remote runner through secrets.
args: dict[str, Any] = json.loads(os.environ.get("MODAL_LAUNCH_ARGS", "{}"))

gradio_app_file = args.get("app_file", "app.py")
if not gradio_app_file.endswith(".py"):
    gradio_app_file = gradio_app_file + ".py"

app = App("modal-gradio-deployment")

image = (
    Image.from_registry(args.get("image"), add_python=args.get("add_python"))
        .pip_install("gradio", "fastapi")
        .env(
            {
                "GRADIO_SERVER_NAME": "0.0.0.0"
            }
        )
        .add_local_file(gradio_app_file, remote_path=f"/root/{gradio_app_file}")
)



volume = (
    Volume.from_name(
        args.get("volume"),
        create_if_missing=True,
    )
    if args.get("volume")
    else None
)
mount_point = args.get("volume_mount", "/root/gradio/volume")
volumes = {mount_point: volume} if volume else {}

@app.function(
    image=image,
    cpu=args.get("cpu"),
    memory=args.get("memory"),
    gpu=args.get("gpu"),
    timeout=args.get("timeout"),
    secrets=[Secret.from_dict({"MODAL_LAUNCH_ARGS": json.dumps(args)})],
    volumes=volumes,
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=7860)
def main():
    # os.makedirs("/root/lab", exist_ok=True)
    token = secrets.token_urlsafe(13)
    # with forward(7860) as tunnel:
    #     url = tunnel.url # + "/?token=" + token
    #     threading.Thread(target=wait_for_port, args=(url, q)).start()
    print("Launching Gradio on Modal, opening in browser...")
    # print(f"   -> {url}\n")
    # q.put(url)
    subprocess.Popen(
        " ".join([
            "python",
            gradio_app_file,
        ]),
        shell=True,
    )
