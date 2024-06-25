# Copyright Modal Labs 2023
import os
import pathlib
import subprocess
import sys
from typing import Optional


def deploy_app_externally(
    servicer,
    file_or_module: str,
    app_variable: Optional[str] = None,
    deployment_name="Deployment",
    cwd=None,
    env={},
    capture_output=True,
) -> Optional[str]:
    # deploys an app from another interpreter to prevent leaking state from client into a container process
    # (apart from what goes through the servicer) also has the advantage that no modules imported by the
    # test files themselves will be added to sys.modules and included in mounts etc.
    windows_support: dict[str, str] = {}

    if sys.platform == "win32":
        windows_support = {
            **os.environ.copy(),
            **{"PYTHONUTF8": "1"},
        }  # windows apparently needs a bunch of env vars to start python...

    env = {**windows_support, "MODAL_SERVER_URL": servicer.client_addr, "MODAL_ENVIRONMENT": "main", **env}
    if cwd is None:
        cwd = pathlib.Path(__file__).parent.parent

    app_ref = file_or_module if app_variable is None else f"{file_or_module}::{app_variable}"

    p = subprocess.Popen(
        [sys.executable, "-m", "modal.cli.entry_point", "deploy", app_ref, "--name", deployment_name],
        cwd=cwd,
        env=env,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE if capture_output else None,
    )
    stdout_b, stderr_b = p.communicate()
    stdout_s, stderr_s = (b.decode() if b is not None else None for b in (stdout_b, stderr_b))
    if p.returncode != 0:
        print(f"Deploying app failed!\n### stdout ###\n{stdout_s}\n### stderr ###\n{stderr_s}")
        raise Exception("Test helper failed to deploy app")
    return stdout_s
