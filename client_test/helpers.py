# Copyright Modal Labs 2023
import pathlib
import subprocess
import sys
from typing import Optional


def deploy_stub_externally(
    servicer, file_or_module: str, stub_variable: Optional[str] = None, deployment_name="Deployment", cwd=None
):
    # deploys a stub from another interpreter to prevent leaking state from client into a container process (apart from what goes through the servicer)
    # also has the advantage that no modules imported by the test files themselves will be added to sys.modules and included in mounts etc.
    env = {"MODAL_SERVER_URL": servicer.remote_addr}
    if cwd is None:
        cwd = pathlib.Path(__file__).parent.parent

    stub_ref = file_or_module if stub_variable is None else f"{file_or_module}::{stub_variable}"

    subprocess.check_call(
        [sys.executable, "-m", "modal.cli.entry_point", "deploy", stub_ref, "--name", deployment_name],
        cwd=cwd,
        env=env,
    )
