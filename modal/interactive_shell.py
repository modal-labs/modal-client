# Copyright Modal Labs 2022
import asyncio
from typing import TYPE_CHECKING, TypeVar

from rich.console import Console

from modal.cli.import_refs import import_function
from modal_utils.async_utils import synchronize_api

from .runner import _run_stub

if TYPE_CHECKING:
    from .stub import _Stub
else:
    _Stub = TypeVar("_Stub")


async def _interactive_shell(_stub: _Stub, cmd: str, environment_name: str = "", func_ref: str = None, **kwargs):
    """Run an interactive shell (like `bash`) within the image for this app.

    This is useful for online debugging and interactive exploration of the
    contents of this image. If `cmd` is optionally provided, it will be run
    instead of the default shell inside this image.

    **Example**

    ```python
    import modal

    stub = modal.Stub(image=modal.Image.debian_slim().apt_install("vim"))
    ```

    You can now run this using

    ```bash
    modal shell script.py --cmd /bin/bash
    ```

    **kwargs will be forwarded to _stub.spawn_sandbox().
    """
    async with _run_stub(_stub, environment_name=environment_name, shell=True):
        function = import_function(func_ref, accept_local_entrypoint=False, accept_webhook=True, base_cmd="modal shell")
        console = Console()
        loading_status = console.status("Starting container...")
        loading_status.start()
        assert function is not None

        asyncio.create_task(function.shell())
        await function.shell()

        # sb = await _stub.spawn_sandbox("sleep", "36000", timeout=3600, **kwargs)

        # for _ in range(40):
        #     await asyncio.sleep(0.5)
        #     resp = await sb._client.stub.SandboxGetTaskId(api_pb2.SandboxGetTaskIdRequest(sandbox_id=sb._object_id))
        #     if resp.task_id != "":
        #         task_id = resp.task_id
        #         break
        #     # else: sandbox hasn't been assigned a task yet
        # else:
        #     print("Error: timed out waiting for sandbox to start")
        #     await sb.terminate()

        # loading_status.stop()
        # await _container_exec(task_id, cmd, tty=True)
        # await sb.terminate()


interactive_shell = synchronize_api(_interactive_shell)
