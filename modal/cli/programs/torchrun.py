import json
import os
import socket
from pathlib import Path
from typing import Any, Dict

import modal

# Passed by `modal launch` locally via CLI, plumbed to remote runner through secrets.
args: Dict[str, Any] = json.loads(os.environ.get("MODAL_LAUNCH_ARGS", "{}"))


def get_i6pn():  # We would provide this or something like it
    """Returns the ipv6 address assigned to this container."""
    return socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0]


def parse_volume_mount(arg: str) -> tuple[str, str]:
    if ":" in arg:
        mnt, name = arg.split(":")
    else:
        mnt, name = f"/mnt/{arg}", arg
    return mnt, modal.Volume.from_name(name)


def get_mount_for_script(script_fname: str) -> modal.Mount:
    parts = Path(script_fname).parts
    local_root = "." if len(parts) == 1 else parts[0]
    return modal.Mount.from_local_dir(local_root, remote_path=Path("/root") / local_root)


cuda_version = args.get("cuda")
base_image = modal.Image.from_registry(f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04", add_python="3.11")
if requirements := args.get("requirements"):
    # Should we parse the requirements and check if we need to install `git`?
    # Note that it will be annoying to have that run consistently locally / remotely
    image = base_image.apt_install("git").pip_install_from_requirements(requirements)
else:
    # Always have at least torch? Anything else?
    image = base_image.pip_install("torch")

app = modal.App()


@app.function(
    cloud="oci",  # TODO remove pending scheduler updates
    gpu=args.get("gpu"),
    timeout=args.get("timeout"),
    cpu=args.get("cpu"),
    memory=args.get("memory"),
    retries=modal.Retries(initial_delay=0.0, max_retries=args.get("retries")),
    image=image,
    mounts=[get_mount_for_script(args.get("script"))],
    volumes=dict(parse_volume_mount(arg) for arg in args.get("volume", [])),
    secrets=[
        *(modal.Secret.from_name(secret) for secret in args.get("secret", [])),
        modal.Secret.from_local_environ(["MODAL_LAUNCH_ARGS"]),
    ],
)
@modal.experimental.grouped(size=args.get("nodes"))
def train(world_size: int, node_rank: int, q: modal.Queue):
    from torch.distributed.run import parse_args, run

    hostname, addr_info = socket.gethostname(), get_i6pn()
    os.environ["NCCL_HOSTID"] = f"{hostname}{addr_info}"
    if node_rank == 0:
        # Broadcast the main address to the other nodes
        q.put_many([addr_info for _ in range(world_size)])
    main_addr = q.get()
    assert main_addr, "Failed to get main i6pn address"
    gpu_proto = modal.gpu.parse_gpu_config(args.get("gpu"))

    run(
        parse_args(
            [
                *(
                    ["--standalone"]
                    if args.get("nodes") == 1
                    else [
                        f"--node-rank={node_rank}",
                        f"--master-addr={main_addr}",
                    ]
                ),
                f"--nnodes={args.get('nodes')}",
                f"--nproc-per-node={gpu_proto.count}",
                "--master-port=1234",
                args.get("script"),
                *args.get("extra_args", []),
            ]
        )
    )


@app.local_entrypoint()
def main():
    handles: list[modal.functions.FunctionCall] = []
    with modal.Queue.ephemeral() as q:
        for node_rank in range(world_size := args.get("nodes")):
            handles.append(train.spawn(world_size, node_rank, q))
        modal.functions.gather(*handles)
