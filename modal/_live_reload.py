# Copyright Modal Labs 2023
import multiprocessing
from multiprocessing.context import SpawnProcess
import platform
import sys
from typing import Optional

from synchronicity import Interface

from modal_utils.async_utils import synchronize_apis, synchronizer

from ._output import OutputManager
from ._watcher import watch
from .app import _App
from .cli.import_refs import import_stub
from .client import _Client
from .stub import StubRunMode


def _run_serve(stub_ref: str, existing_app_id: str):
    # subprocess entrypoint
    _stub = import_stub(stub_ref)
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)
    blocking_stub.serve(existing_app_id=existing_app_id)


def restart_serve(stub_ref: str, existing_app_id: str, prev_proc: Optional[SpawnProcess]) -> SpawnProcess:
    if prev_proc is not None:
        prev_proc.terminate()
    ctx = multiprocessing.get_context("spawn")  # Needed to reload the interpreter
    p = ctx.Process(target=_run_serve, args=(stub_ref, existing_app_id))
    p.start()
    return p


async def _run_serve_loop(stub_ref: str, timeout: Optional[float] = None, stdout=None, show_progress=True):
    stub = import_stub(stub_ref)

    unsupported_msg = None
    if platform.system() == "Windows":
        unsupported_msg = "Live-reload skipped. This feature is currently unsupported on Windows"
        " This can hopefully be fixed in a future version of Modal."
    elif sys.version_info < (3, 8):
        unsupported_msg = (
            "Live-reload skipped. This feature is unsupported below Python 3.8."
            " Upgrade to Python 3.8+ to enable live-reloading."
        )

    client = await _Client.from_env()

    output_mgr = OutputManager(stdout, show_progress)

    if unsupported_msg:
        async with stub._run(client, output_mgr, None, mode=StubRunMode.SERVE) as app:
            client.set_pre_stop(app.disconnect)
            async for _ in watch(stub._local_mounts, output_mgr, timeout):
                output_mgr.print_if_visible(unsupported_msg)
    else:
        app = await _App._init_new(client, stub.description, deploying=False, detach=False)
        curr_proc = None
        try:
            async for _ in watch(stub._local_mounts, output_mgr, timeout):
                curr_proc = restart_serve(stub_ref, existing_app_id=app.app_id, prev_proc=curr_proc)
        finally:
            if curr_proc:
                curr_proc.terminate()


run_serve_loop, aio_run_serve_loop = synchronize_apis(_run_serve_loop)
