# Copyright Modal Labs 2023
import os
import pathlib
import signal
import subprocess
import sys
from typing import Optional, Tuple


def deploy_app_externally(
    servicer,
    credentials: Tuple[str, str],
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

    token_id, token_secret = credentials
    env = {
        **windows_support,
        "MODAL_SERVER_URL": servicer.client_addr,
        "MODAL_TOKEN_ID": token_id,
        "MODAL_TOKEN_SECRET": token_secret,
        "MODAL_ENVIRONMENT": "main",
        **env,
    }
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


class PopenWithCtrlC(subprocess.Popen):
    def __init__(self, *args, creationflags=0, **kwargs):
        if sys.platform == "win32":
            # needed on windows to separate ctrl-c lifecycle of subprocess from parent:
            creationflags = creationflags | subprocess.CREATE_NEW_CONSOLE  # type: ignore

        super().__init__(*args, **kwargs, creationflags=creationflags)

    def send_ctrl_c(self):
        # platform independent way to replicate the behavior of Ctrl-C:ing a cli app
        if sys.platform == "win32":
            # windows doesn't support sigint, and subprocess.CTRL_C_EVENT has a bunch
            # of gotchas since it's bound to a console which is the same for the parent
            # process by default, and can't be sent using the python standard library
            # to a separate process's console
            import console_ctrl

            console_ctrl.send_ctrl_c(self.pid)  # noqa [E731]
        else:
            self.send_signal(signal.SIGINT)
