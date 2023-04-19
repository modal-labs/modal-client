# Copyright Modal Labs 2023
import contextlib
import io
import multiprocessing
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event
import platform
import sys
from typing import AsyncGenerator, Optional

from synchronicity import Interface

from modal_utils.async_utils import TaskContext, asyncify, synchronize_apis, synchronizer

from ._output import OutputManager
from ._watcher import watch
from .app import _App
from .cli.import_refs import import_stub
from .client import _Client
from .runner import run_stub


def _run_serve(stub_ref: str, existing_app_id: str, is_ready: Event):
    # subprocess entrypoint
    _stub = import_stub(stub_ref)
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)
    blocking_stub.serve_update(existing_app_id, is_ready)


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


async def _run_watch_loop(stub_ref: str, app_id: str, output_mgr: OutputManager, watcher: AsyncGenerator[None, None]):
    unsupported_msg = None
    if platform.system() == "Windows":
        unsupported_msg = "Live-reload skipped. This feature is currently unsupported on Windows"
        " This can hopefully be fixed in a future version of Modal."
    elif sys.version_info < (3, 8):
        unsupported_msg = (
            "Live-reload skipped. This feature is unsupported below Python 3.8."
            " Upgrade to Python 3.8+ to enable live-reloading."
        )

    if unsupported_msg:
        async for _ in watcher:
            output_mgr.print_if_visible(unsupported_msg)
    else:
        curr_proc = None
        try:
            async for _ in watcher:
                await _terminate(curr_proc, output_mgr)
                curr_proc = await _restart_serve(stub_ref, existing_app_id=app_id)
        finally:
            await _terminate(curr_proc, output_mgr)


def _get_clean_stub_description(stub_ref: str) -> str:
    # If possible, consider the 'ref' argument the start of the app's args. Everything
    # before it Modal CLI cruft (eg. `modal serve --timeout 1.0`).
    try:
        func_ref_arg_idx = sys.argv.index(stub_ref)
        return " ".join(sys.argv[func_ref_arg_idx:])
    except ValueError:
        return " ".join(sys.argv)


@contextlib.asynccontextmanager
async def _serve_stub(
    stub_ref: str,
    stdout: Optional[io.TextIOWrapper] = None,
    show_progress: bool = True,
    _watcher: Optional[AsyncGenerator[None, None]] = None,  # for testing
) -> AsyncGenerator[_App, None]:
    stub = import_stub(stub_ref)
    if stub._description is None:
        stub._description = _get_clean_stub_description(stub_ref)

    client = await _Client.from_env()

    output_mgr = OutputManager(stdout, show_progress, "Running app...")

    if _watcher is not None:
        watcher = _watcher  # Only used by tests
    else:
        watcher = watch(stub._local_mounts, output_mgr)

    async with run_stub(stub, client=client, output_mgr=output_mgr) as app:
        client.set_pre_stop(app.disconnect)
        async with TaskContext(grace=0.1) as tc:
            tc.create_task(_run_watch_loop(stub_ref, app.app_id, output_mgr, watcher))
            yield app


serve_stub, aio_serve_stub = synchronize_apis(_serve_stub)
