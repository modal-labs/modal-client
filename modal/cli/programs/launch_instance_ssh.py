# Copyright Modal Labs 2023
# type: ignore
import json
import os
import sys
from typing import Any

import rich
import rich.panel
import rich.rule

import modal
import modal.experimental

# Passed by `modal launch` locally via CLI.
args: dict[str, Any] = json.loads(os.environ.get("MODAL_LAUNCH_ARGS", "{}"))

app = modal.App()

image: modal.Image
if args.get("image"):
    image = modal.Image.from_registry(args.get("image"))
else:
    # Must be set to the same image builder version as the notebook base image.
    os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2024.10"
    image = modal.experimental.notebook_base_image(python_version="3.12")

volume = (
    modal.Volume.from_name(
        args.get("volume"),
        create_if_missing=True,
    )
    if args.get("volume")
    else None
)
volumes = {"/workspace": volume} if volume else {}


startup_script = """
set -eu
mkdir -p /run/sshd

# Check if sshd is installed, install if not
test -x /usr/sbin/sshd || (apt-get update && apt-get install -y openssh-server)

# Change default working directory to /workspace
echo "cd /workspace" >> /root/.profile

mkdir -p /root/.ssh
echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
/usr/sbin/sshd -D -e
"""


@app.local_entrypoint()
def main():
    if not os.environ.get("SSH_PUBLIC_KEY"):
        raise ValueError("SSH_PUBLIC_KEY environment variable is not set")

    sb = modal.Sandbox.create(
        *("sh", "-c", startup_script),
        app=app,
        image=image,
        cpu=args.get("cpu"),
        memory=args.get("memory"),
        gpu=args.get("gpu"),
        timeout=args.get("timeout"),
        volumes=volumes,
        unencrypted_ports=[22],  # Forward SSH port
        secrets=[modal.Secret.from_dict({"SSH_PUBLIC_KEY": os.environ.get("SSH_PUBLIC_KEY")})],
    )
    hostname, port = sb.tunnels()[22].tcp_socket
    connection_cmd = f"ssh -A -p {port} root@{hostname}"

    rich.print(
        rich.rule.Rule(style="yellow"),
        rich.panel.Panel(
            f"""Your instance is ready! You can SSH into it using the following command:

  [dim gray]>[/dim gray] [bold cyan]{connection_cmd}[/bold cyan]

[italic]Details:[/italic]
  • Name:    [magenta]{app.description}[/magenta]
  • CPU:     [yellow]{args.get("cpu")} cores[/yellow]
  • Memory:  [yellow]{args.get("memory")} MiB[/yellow]
  • Timeout: [yellow]{args.get("timeout")} seconds[/yellow]
  • GPU:     [green]{(args.get("gpu") or "N/A").upper()}[/green]""",
            title="SSH Connection",
            expand=False,
        ),
        rich.rule.Rule(style="yellow"),
    )

    sys.exit(0)  # Exit immediately to prevent "Timed out waiting for final apps log."
