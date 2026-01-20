# Copyright Modal Labs 2022
"""Output management interface for Modal CLI.

This module defines the interface for output management and provides a no-op
implementation for when output is disabled. The rich-based implementation lives
in _rich_output.py to avoid importing rich when it's not needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import sys
from collections.abc import Generator
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, runtime_checkable

from grpclib.exceptions import StreamTerminatedError

from modal_proto import api_pb2

from ._utils.grpc_utils import Retry
from ._utils.shell_utils import stream_from_stdin, write_to_fd
from .config import logger
from .exception import InternalError, ServiceError

if TYPE_CHECKING:
    from .client import _Client


@runtime_checkable
class StatusRow(Protocol):
    """Protocol describing a row in the object creation status tree."""

    def message(self, message: str) -> None:
        """Update the primary message shown for this row."""
        ...

    def warning(self, warning: "api_pb2.Warning") -> None:
        """Append a warning message associated with this row."""
        ...

    def finish(self, message: str) -> None:
        """Mark the row as finished with the given message."""
        ...


class DisabledStatusRow:
    """No-op StatusRow used when output is disabled."""

    def message(self, message: str) -> None:
        pass

    def warning(self, warning: "api_pb2.Warning") -> None:
        pass

    def finish(self, message: str) -> None:
        pass


@runtime_checkable
class OutputManager(Protocol):
    """Protocol defining the interface for output management.

    This protocol allows for different implementations:
    - RichOutputManager: Full rich-based terminal output with progress spinners, trees, etc.
    - DisabledOutputManager: No-op implementation for when output is disabled.

    Using a protocol allows code to work with any output manager without checking for None.
    """

    @property
    def is_enabled(self) -> bool:
        """Whether rich output is enabled."""
        ...

    @property
    def _stdout(self) -> Any:
        """The stdout stream for PTY shell output."""
        ...

    @property
    def _show_image_logs(self) -> bool:
        """Whether to show image logs."""
        ...

    def disable(self) -> None:
        """Disable output and clean up resources."""
        ...

    @contextlib.contextmanager
    def display_object_tree(self) -> Generator[None, None, None]:
        """Context manager that displays a tree of objects being created."""
        ...

    def add_status_row(self) -> StatusRow:
        """Add a status row to the current object tree."""
        ...

    def print(self, renderable: Any) -> None:
        """Print a renderable to the console."""
        ...

    def make_live(self, renderable: Any) -> AbstractContextManager[Any]:
        """Create a Live context manager for the given renderable."""
        ...

    def enable_image_logs(self) -> None:
        """Enable showing image logs."""
        ...

    def show_status_spinner(self) -> AbstractContextManager[None]:
        """Context manager that shows a status spinner."""
        ...

    def update_app_page_url(self, app_page_url: str) -> None:
        """Update the app page URL for display."""
        ...

    def function_progress_callback(self, tag: str, total: int | None) -> Callable[[int, int], None]:
        """Get a callback for updating function progress."""
        ...

    def update_task_state(self, task_id: str, state: int) -> None:
        """Update the state of a task."""
        ...

    def update_snapshot_progress(self, image_id: str, task_progress: "api_pb2.TaskProgress") -> None:
        """Update snapshot upload progress."""
        ...

    def update_queueing_progress(
        self, *, function_id: str, completed: int, total: int | None, description: str | None
    ) -> None:
        """Update function queueing progress."""
        ...

    async def put_log_content(self, log: "api_pb2.TaskLogs") -> None:
        """Process and display log content."""
        ...

    def flush_lines(self) -> None:
        """Flush any buffered output."""
        ...

    @staticmethod
    def step_progress(text: str = "") -> Any:
        """Returns the element to be rendered when a step is in progress."""
        ...

    @staticmethod
    def step_completed(message: str) -> Any:
        """Returns the element to be rendered when a step is completed."""
        ...

    @staticmethod
    def substep_completed(message: str) -> Any:
        """Returns the element to be rendered when a substep is completed."""
        ...


class DisabledOutputManager:
    """No-op implementation of OutputManager for when output is disabled.

    All methods are no-ops that do nothing, allowing code to call output methods
    without checking if the output manager exists.
    """

    @property
    def is_enabled(self) -> bool:
        return False

    @property
    def _stdout(self) -> Any:
        return sys.stdout

    @property
    def _show_image_logs(self) -> bool:
        return True  # Always "show" logs when disabled (don't filter them)

    def disable(self) -> None:
        pass

    @contextlib.contextmanager
    def display_object_tree(self) -> Generator[None, None, None]:
        yield

    def add_status_row(self) -> StatusRow:
        return DisabledStatusRow()

    def print(self, renderable: Any) -> None:
        pass

    def make_live(self, renderable: Any) -> AbstractContextManager[Any]:
        return nullcontext()

    def enable_image_logs(self) -> None:
        pass

    @contextlib.contextmanager
    def show_status_spinner(self) -> Generator[None, None, None]:
        yield

    def update_app_page_url(self, app_page_url: str) -> None:
        pass

    def function_progress_callback(self, tag: str, total: int | None) -> Callable[[int, int], None]:
        def noop(completed: int, total: int) -> None:
            pass

        return noop

    def update_task_state(self, task_id: str, state: int) -> None:
        pass

    def update_snapshot_progress(self, image_id: str, task_progress: "api_pb2.TaskProgress") -> None:
        pass

    def update_queueing_progress(
        self, *, function_id: str, completed: int, total: int | None, description: str | None
    ) -> None:
        pass

    async def put_log_content(self, log: "api_pb2.TaskLogs") -> None:
        pass

    def flush_lines(self) -> None:
        pass

    @staticmethod
    def step_progress(text: str = "") -> str:
        return text

    @staticmethod
    def step_completed(message: str) -> str:
        return message

    @staticmethod
    def substep_completed(message: str) -> str:
        return message


# Singleton instance of the disabled output manager
_DISABLED_OUTPUT_MANAGER = DisabledOutputManager()


def _get_suffix_from_web_url_info(url_info: api_pb2.WebUrlInfo) -> str:
    if url_info.truncated:
        suffix = " [grey70](label truncated)[/grey70]"
    elif url_info.label_stolen:
        suffix = " [grey70](label stolen)[/grey70]"
    else:
        suffix = ""
    return suffix


class FunctionCreationStatus:
    """Context manager for tracking and displaying function creation progress."""

    tag: str
    response: Optional[api_pb2.FunctionCreateResponse] = None

    def __init__(self, tag: str):
        from modal.output import _get_output_manager

        self.tag = tag
        self._output_mgr = _get_output_manager()

    def __enter__(self):
        self.status_row = self._output_mgr.add_status_row()
        self.status_row.message(f"Creating function {self.tag}...")
        return self

    def set_response(self, resp: api_pb2.FunctionCreateResponse):
        self.response = resp

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            raise exc_val

        if not self.response:
            self.status_row.finish(f"Unknown error when creating function {self.tag}")

        elif web_url := self.response.handle_metadata.web_url:
            url_info = self.response.function.web_url_info
            requires_proxy_auth = self.response.function.webhook_config.requires_proxy_auth
            proxy_auth_suffix = " 🔑" if requires_proxy_auth else ""
            # Ensure terms used here match terms used in modal.com/docs/guide/webhook-urls doc.
            suffix = _get_suffix_from_web_url_info(url_info)
            # TODO: this is only printed when we're showing progress. Maybe move this somewhere else.
            for warning in self.response.server_warnings:
                self.status_row.warning(warning)
            self.status_row.finish(
                f"Created web function {self.tag} => [magenta underline]{web_url}[/magenta underline]"
                f"{proxy_auth_suffix}{suffix}"
            )

            # Print custom domain in terminal
            for custom_domain in self.response.function.custom_domain_info:
                custom_domain_status_row = self._output_mgr.add_status_row()
                custom_domain_status_row.finish(
                    f"Custom domain for {self.tag} => [magenta underline]{custom_domain.url}[/magenta underline]"
                )

        elif self.response.function.flash_service_urls:
            for flash_service_url in self.response.function.flash_service_urls:
                flash_service_url_status_row = self._output_mgr.add_status_row()
                flash_service_url_status_row.finish(
                    f"Created flash service endpoint for {self.tag} => "
                    f"[magenta underline]{flash_service_url}[/magenta underline]"
                )

        else:
            for warning in self.response.server_warnings:
                self.status_row.warning(warning)
            self.status_row.finish(f"Created function {self.tag}.")
            if self.response.function.method_definitions_set:
                for method_definition in self.response.function.method_definitions.values():
                    if method_definition.web_url:
                        url_info = method_definition.web_url_info
                        suffix = _get_suffix_from_web_url_info(url_info)
                        class_web_endpoint_method_status_row = self._output_mgr.add_status_row()
                        class_web_endpoint_method_status_row.finish(
                            f"Created web endpoint for {method_definition.function_name} => [magenta underline]"
                            f"{method_definition.web_url}[/magenta underline]{suffix}"
                        )
                        for custom_domain in method_definition.custom_domain_info:
                            custom_domain_status_row = self._output_mgr.add_status_row()
                            custom_domain_status_row.finish(
                                f"Custom domain for {method_definition.function_name} => [magenta underline]"
                                f"{custom_domain.url}[/magenta underline]"
                            )


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


async def put_pty_content(log: api_pb2.TaskLogs, stdout):
    if hasattr(stdout, "buffer"):
        # If we're not showing progress, there's no need to buffer lines,
        # because the progress spinner can't interfere with output.

        data = log.data.encode("utf-8")
        # Non-blocking terminals can fill the kernel buffer on output bursts, making flush() raise
        # BlockingIOError (EAGAIN) and appear frozen until a key is pressed (this happened e.g. when
        # printing large data from a pdb breakpoint). If stdout has a real fd, we await a
        # non-blocking fd write (write_to_fd) instead.
        fd = None
        try:
            if hasattr(stdout, "fileno"):
                fd = stdout.fileno()
        except Exception:
            fd = None

        if fd is not None:
            await write_to_fd(fd, data)
        else:
            # For streams without fileno(), use the normal write/flush path.
            written = 0
            n_retries = 0
            while written < len(data):
                try:
                    written += stdout.buffer.write(data[written:])
                    stdout.flush()
                except BlockingIOError:
                    if n_retries >= 5:
                        raise
                    n_retries += 1
                    await asyncio.sleep(0.1)
    else:
        # `stdout` isn't always buffered (e.g. %%capture in Jupyter notebooks redirects it to
        # io.StringIO).
        stdout.write(log.data)
        stdout.flush()


async def get_app_logs_loop(
    client: "_Client",
    output_mgr: OutputManager,
    app_id: str | None = None,
    task_id: str | None = None,
    app_logs_url: str | None = None,
):
    last_log_batch_entry_id = ""

    pty_shell_stdout = None
    pty_shell_finish_event: asyncio.Event | None = None
    pty_shell_task_id: str | None = None
    pty_shell_input_task: asyncio.Task | None = None

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
                await put_pty_content(log, pty_shell_stdout)
            else:
                await output_mgr.put_log_content(log)

    async def _get_logs():
        nonlocal last_log_batch_entry_id
        nonlocal pty_shell_stdout, pty_shell_finish_event, pty_shell_task_id, pty_shell_input_task

        request = api_pb2.AppGetLogsRequest(
            app_id=app_id or "",
            task_id=task_id or "",
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
                    pty_shell_stdout = output_mgr._stdout
                    pty_shell_finish_event = asyncio.Event()
                    pty_shell_task_id = log_batch.task_id
                    output_mgr.disable()
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
