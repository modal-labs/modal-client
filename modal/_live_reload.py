# Copyright Modal Labs 2023
import asyncio
import io
import multiprocessing
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event
import platform
import sys
from typing import AsyncGenerator, Optional

from synchronicity import Interface

from modal_utils.async_utils import asyncify, synchronize_apis, synchronizer

from ._output import OutputManager, step_progress
from ._watcher import watch
from .app import _App
from .cli.import_refs import import_stub
from .client import _Client


def _run_serve(stub_ref: str, existing_app_id: str, is_ready: Event):
    # subprocess entrypoint
    _stub = import_stub(stub_ref)
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)
    blocking_stub._serve_update(existing_app_id, is_ready)


async def _restart_serve(stub_ref: str, existing_app_id: str, timeout: float = 5.0) -> SpawnProcess:
    ctx = multiprocessing.get_context("spawn")  # Needed to reload the interpreter
    is_ready = ctx.Event()
    p = ctx.Process(target=_run_serve, args=(stub_ref, existing_app_id, is_ready))
    p.start()
    await asyncify(is_ready.wait)(timeout)
    # TODO(erikbern): we don't fail if the above times out, but that's somewhat intentional, since
    # the child process might build a huge image or similar
    return p


async def _terminate(proc: Optional[SpawnProcess], output_mgr: OutputManager, timeout: float = 5.0):
    if proc is None:
        return
    try:
        proc.terminate()
        await asyncify(proc.join)(timeout)
        if proc.exitcode is not None:
            output_mgr.print_if_visible(f"Serve process {proc.pid} terminated")
        else:
            output_mgr.print_if_visible(
                f"[red]Serve process {proc.pid} didn't terminate after {timeout}s, killing it[/red]"
            )
            proc.kill()
    except ProcessLookupError:
        pass  # Child process already finished


async def _run_serve_loop(
    stub_ref: str,
    timeout: Optional[float] = None,
    stdout: Optional[io.TextIOWrapper] = None,
    show_progress: bool = True,
    _watcher: Optional[AsyncGenerator[None, None]] = None,  # for testing
    _app_q: Optional[asyncio.Queue] = None,  # for testing
):
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

    if _watcher is not None:
        watcher = _watcher  # Only used by tests
    else:
        watcher = watch(stub._local_mounts, output_mgr, timeout)

    app = await _App._init_new(client, stub.description, detach=False, deploying=False)
    status_spinner = step_progress("Running app...")

    if unsupported_msg:
        async with stub._run(client, output_mgr, app, status_spinner=status_spinner):
            client.set_pre_stop(app.disconnect)
            async for _ in watcher:
                output_mgr.print_if_visible(unsupported_msg)
    else:
        # Run the object creation loop one time first, to make sure all images etc get built
        # This also handles the logs and the heartbeats
        async with stub._run(client, output_mgr, app, status_spinner=status_spinner):
            if _app_q:
                await _app_q.put(app)
            client.set_pre_stop(app.disconnect)
            existing_app_id = app.app_id
            curr_proc = None
            try:
                async for _ in watcher:
                    await _terminate(curr_proc, output_mgr)
                    curr_proc = await _restart_serve(stub_ref, existing_app_id=existing_app_id)
            finally:
                await _terminate(curr_proc, output_mgr)


run_serve_loop, aio_run_serve_loop = synchronize_apis(_run_serve_loop)
