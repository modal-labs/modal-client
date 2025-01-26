# Copyright Modal Labs 2025
import json
import os
import time

import modal

# Passed by `modal launch` locally via CLI, plumbed to remote runner through secrets.
args: dict = json.loads(os.environ.get("MODAL_LAUNCH_ARGS", "{}"))

CACHE_DIR = "/hf-cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub[hf-transfer]==0.27.1")
    .env({"HF_HUB_CACHE": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
volume = modal.Volume.from_name(str(args.get("volume")))
secrets = [modal.Secret.from_dict({"MODAL_LAUNCH_ARGS": json.dumps(args)})]
if user_secret := args.get("secret"):
    secrets.append(modal.Secret.from_name(user_secret))
app = modal.App("hf-download", image=image, secrets=secrets, volumes={CACHE_DIR: volume})


@app.function(cpu=4, memory=1028, timeout=int(args.get("timeout", 600)))
def run():
    from huggingface_hub import snapshot_download

    t0 = time.monotonic()
    snapshot_download(
        repo_id=args.get("repo_id"),
        repo_type=args.get("type"),
        revision=args.get("revision"),
        ignore_patterns=args.get("ignore", []),
        allow_patterns=args.get("allow", []),
        cache_dir=CACHE_DIR,
    )
    print(f"Completed in {time.monotonic() - t0:.2f} seconds")


@app.local_entrypoint()
def main():
    run.remote()
