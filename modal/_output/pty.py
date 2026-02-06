# Copyright Modal Labs 2022
"""PTY utilities and log streaming for Modal CLI.

This module contains PTY/terminal handling utilities and log streaming functions
that handle interactive terminal sessions and application log output.
"""

import asyncio
import os
import socket
from typing import TYPE_CHECKING, Optional

from grpclib.exceptions import StreamTerminatedError

from modal._utils.grpc_utils import Retry
from modal._utils.shell_utils import get_winsz, stream_from_stdin
from modal.config import logger
from modal.exception import InternalError, ServiceError
from modal_proto import api_pb2

if TYPE_CHECKING:
    from modal._output.manager import OutputManager
    from modal.client import _Client


# =============================================================================
# PTY Utilities
# =============================================================================


def get_pty_info(shell: bool, no_terminate_on_idle_stdin: bool = False) -> api_pb2.PTYInfo:
    rows, cols = get_winsz()
    return api_pb2.PTYInfo(
        enabled=True,  # TODO(erikbern): deprecated
        winsz_rows=rows,
        winsz_cols=cols,
        env_term=os.environ.get("TERM"),
        env_colorterm=os.environ.get("COLORTERM"),
        env_term_program=os.environ.get("TERM_PROGRAM"),
        pty_type=api_pb2.PTYInfo.PTY_TYPE_SHELL if shell else api_pb2.PTYInfo.PTY_TYPE_FUNCTION,
        no_terminate_on_idle_stdin=no_terminate_on_idle_stdin,
    )


# =============================================================================
# Log Streaming
# =============================================================================


async def stream_pty_shell_input(client: "_Client", exec_id: str, finish_event: asyncio.Event):
    """
    Streams stdin to the given exec id until finish_event is triggered
    """

    async def _handle_input(data: bytes, message_index: int):
        await client.stub.ContainerExecPutInput(
            api_pb2.ContainerExecPutInputRequest(
                exec_id=exec_id, input=api_pb2.RuntimeInputMessage(message=data, message_index=message_index)
            ),
            retry=Retry(total_timeout=10),
        )

    async with stream_from_stdin(_handle_input, use_raw_terminal=True):
        await finish_event.wait()


async def get_app_logs_loop(
    client: "_Client",
    output_mgr: "OutputManager",
    app_id: Optional[str] = None,
    task_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
):
    last_log_batch_entry_id = ""

    pty_shell_finish_event: Optional[asyncio.Event] = None
    pty_shell_task_id: Optional[str] = None
    pty_shell_input_task: Optional[asyncio.Task] = None

    async def stop_pty_shell():
        nonlocal pty_shell_finish_event, pty_shell_input_task
        if pty_shell_finish_event:
            print("\r", end="")  # move cursor to beginning of line # noqa: T201
            pty_shell_finish_event.set()
            pty_shell_finish_event = None

            if pty_shell_input_task:
                try:
                    await pty_shell_input_task
                except Exception as exc:
                    logger.exception(f"Exception in PTY shell input task: {exc}")
                finally:
                    pty_shell_input_task = None

    async def _put_log(log_batch: api_pb2.TaskLogsBatch, log: api_pb2.TaskLogs):
        if log.task_state:
            output_mgr.update_task_state(log_batch.task_id, log.task_state)
            if log.task_state == api_pb2.TASK_STATE_WORKER_ASSIGNED:
                # Close function's queueing progress bar (if it exists)
                output_mgr.update_queueing_progress(
                    function_id=log_batch.function_id, completed=1, total=1, description=None
                )
        elif log.task_progress.len or log.task_progress.pos:
            if log.task_progress.progress_type == api_pb2.FUNCTION_QUEUED:
                output_mgr.update_queueing_progress(
                    function_id=log_batch.function_id,
                    completed=log.task_progress.pos,
                    total=log.task_progress.len,
                    description=log.task_progress.description,
                )
            else:  # Ensure forward-compatible with new types.
                logger.debug(f"Received unrecognized progress type: {log.task_progress.progress_type}")
        elif log.data:
            if pty_shell_finish_event:
                await output_mgr.put_pty_content(log)
            else:
                await output_mgr.put_log_content(log)

    async def _get_logs():
        nonlocal last_log_batch_entry_id
        nonlocal pty_shell_finish_event, pty_shell_task_id, pty_shell_input_task

        request = api_pb2.AppGetLogsRequest(
            app_id=app_id or "",
            task_id=task_id or "",
            sandbox_id=sandbox_id or "",
            timeout=55,
            last_entry_id=last_log_batch_entry_id,
        )
        log_batch: api_pb2.TaskLogsBatch
        async for log_batch in client.stub.AppGetLogs.unary_stream(request):
            if log_batch.entry_id:
                # log_batch entry_id is empty for fd="server" messages from AppGetLogs
                last_log_batch_entry_id = log_batch.entry_id
            if log_batch.app_done:
                logger.debug("App logs are done")
                last_log_batch_entry_id = None
                break
            elif log_batch.image_id and not output_mgr._show_image_logs:
                # Ignore image logs while app is creating objects.
                # These logs are fetched through ImageJoinStreaming instead.
                # Logs from images built "dynamically" (after the app has started)
                # are printed through this loop.
                # TODO (akshat): have a better way of differentiating between
                # statically and dynamically built images.
                pass
            elif log_batch.pty_exec_id:
                # This corresponds to the `modal run -i` use case where a breakpoint
                # triggers and the task drops into an interactive PTY mode
                if pty_shell_finish_event:
                    print("ERROR: concurrent PTY shells are not supported.")  # noqa: T201
                else:
                    pty_shell_finish_event = asyncio.Event()
                    pty_shell_task_id = log_batch.task_id
                    # Clean up output state before entering PTY mode
                    output_mgr.flush_lines()
                    output_mgr.stop_status_spinner()
                    output_mgr.set_quiet_mode(True)
                    pty_shell_input_task = asyncio.create_task(
                        stream_pty_shell_input(client, log_batch.pty_exec_id, pty_shell_finish_event)
                    )
            else:
                for log in log_batch.items:
                    await _put_log(log_batch, log)

            if log_batch.eof and log_batch.task_id == pty_shell_task_id:
                await stop_pty_shell()

        output_mgr.flush_lines()

    while True:
        try:
            await _get_logs()
        except (ServiceError, InternalError, StreamTerminatedError, socket.gaierror, AttributeError) as exc:
            if isinstance(exc, (ServiceError, InternalError)):
                # Try again if we had a temporary connection drop, for example if computer went to sleep.
                logger.debug("Log fetching timed out. Retrying ...")
                continue
            elif isinstance(exc, StreamTerminatedError):
                logger.debug("Stream closed. Retrying ...")
                continue
            elif isinstance(exc, socket.gaierror):
                logger.debug("Lost connection. Retrying ...")
                continue
            elif isinstance(exc, AttributeError):
                if "_write_appdata" in str(exc):
                    # Happens after losing connection
                    # StreamTerminatedError are not properly raised in grpclib<=0.4.7
                    # fixed in https://github.com/vmagamedov/grpclib/issues/185
                    # TODO: update to newer version (>=0.4.8) once stable
                    logger.debug("Lost connection. Retrying ...")
                    continue
            raise

        if last_log_batch_entry_id is None:
            break

    await stop_pty_shell()

    logger.debug("Logging exited gracefully")
