# Copyright Modal Labs 2023
import multiprocessing
import platform
import sys
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, AsyncGenerator, Optional, Set, TypeVar

from synchronicity import Interface
from synchronicity.async_wrap import asynccontextmanager

from ._utils.async_utils import TaskContext, asyncify, synchronize_api, synchronizer
from ._utils.logger import logger
from ._watcher import watch
from .cli.import_refs import import_app
from .client import _Client
from .config import config
from .exception import deprecation_error
from .output import _get_output_manager
from .runner import _run_app, serve_update

if TYPE_CHECKING:
    from .app import _App
else:
    _App = TypeVar("_App")


def _run_serve(app_ref: str, existing_app_id: str, is_ready: Event, environment_name: str):
    # subprocess entrypoint
    _app = import_app(app_ref)
    blocking_app = synchronizer._translate_out(_app, Interface.BLOCKING)
    serve_update(blocking_app, existing_app_id, is_ready, environment_name)


async def _restart_serve(
    app_ref: str, existing_app_id: str, environment_name: str, timeout: float = 5.0
) -> SpawnProcess:
    ctx = multiprocessing.get_context("spawn")  # Needed to reload the interpreter
    is_ready = ctx.Event()
    p = ctx.Process(target=_run_serve, args=(app_ref, existing_app_id, is_ready, environment_name))
    p.start()
    await asyncify(is_ready.wait)(timeout)
    # TODO(erikbern): we don't fail if the above times out, but that's somewhat intentional, since
    # the child process might build a huge image or similar
    return p


async def _terminate(proc: Optional[SpawnProcess], timeout: float = 5.0):
    if proc is None:
        return
    try:
        proc.terminate()
        await asyncify(proc.join)(timeout)
        if proc.exitcode is not None:
            if output_mgr := _get_output_manager():
                output_mgr.print(f"Serve process {proc.pid} terminated")
        else:
            if output_mgr := _get_output_manager():
                output_mgr.print(f"[red]Serve process {proc.pid} didn't terminate after {timeout}s, killing it[/red]")
            proc.kill()
    except ProcessLookupError:
        pass  # Child process already finished


async def _run_watch_loop(
    app_ref: str,
    app_id: str,
    watcher: AsyncGenerator[Set[str], None],
    environment_name: str,
):
    unsupported_msg = None
    if platform.system() == "Windows":
        unsupported_msg = "Live-reload skipped. This feature is currently unsupported on Windows"
        " This can hopefully be fixed in a future version of Modal."

    if unsupported_msg:
        if output_mgr := _get_output_manager():
            async for _ in watcher:
                output_mgr.print(unsupported_msg)
    else:
        curr_proc = None
        try:
            async for trigger_files in watcher:
                logger.debug(f"The following files triggered an app update: {', '.join(trigger_files)}")
                await _terminate(curr_proc)
                curr_proc = await _restart_serve(app_ref, existing_app_id=app_id, environment_name=environment_name)
        finally:
            await _terminate(curr_proc)


def _get_clean_app_description(app_ref: str) -> str:
    # If possible, consider the 'ref' argument the start of the app's args. Everything
    # before it Modal CLI cruft (eg. `modal serve --timeout 1.0`).
    try:
        func_ref_arg_idx = sys.argv.index(app_ref)
        return " ".join(sys.argv[func_ref_arg_idx:])
    except ValueError:
        return " ".join(sys.argv)


@asynccontextmanager
async def _serve_app(
    app: "_App",
    app_ref: str,
    _watcher: Optional[AsyncGenerator[Set[str], None]] = None,  # for testing
    environment_name: Optional[str] = None,
) -> AsyncGenerator["_App", None]:
    if environment_name is None:
        environment_name = config.get("environment")

    client = await _Client.from_env()

    if _watcher is not None:
        watcher = _watcher  # Only used by tests
    else:
        mounts_to_watch = app._get_watch_mounts()
        watcher = watch(mounts_to_watch)

    async with _run_app(app, client=client, environment_name=environment_name):
        async with TaskContext(grace=0.1) as tc:
            tc.create_task(_run_watch_loop(app_ref, app.app_id, watcher, environment_name))
            yield app


def _serve_stub(*args, **kwargs):
    deprecation_error((2024, 5, 1), "`serve_stub` is deprecated. Please use `serve_app` instead.")


serve_app = synchronize_api(_serve_app)
serve_stub = synchronize_api(_serve_stub)
