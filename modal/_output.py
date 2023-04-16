# Copyright Modal Labs 2022
from __future__ import annotations
import asyncio
import contextlib
import functools
import io
import platform
import re
import sys
from datetime import timedelta
from typing import Callable, Optional

from grpclib.exceptions import GRPCError, StreamTerminatedError
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.spinner import Spinner
from rich.text import Text

from modal_proto import api_pb2
from modal_utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, unary_stream

from ._ipython import is_notebook
from .client import _Client
from .config import logger

if platform.system() == "Windows":
    default_spinner = "line"
else:
    default_spinner = "dots"


class FunctionQueuingColumn(ProgressColumn):
    """Renders time elapsed, including task.completed as additional elapsed time."""

    def __init__(self):
        self.lag = 0
        super().__init__()

    def render(self, task) -> Text:
        self.lag = max(task.completed - task.elapsed, self.lag)
        if task.finished:
            elapsed = max(task.finished_time, task.completed)
        else:
            elapsed = task.elapsed + self.lag
        delta = timedelta(seconds=int(elapsed))
        return Text(str(delta), style="progress.elapsed")


def step_progress(text: str = "") -> Spinner:
    """Returns the element to be rendered when a step is in progress."""
    return Spinner(default_spinner, text, style="blue")


def step_progress_update(spinner: Spinner, message: str):
    spinner.update(text=message)


def step_completed(message: str, is_substep: bool = False) -> RenderableType:
    """Returns the element to be rendered when a step is completed."""

    STEP_COMPLETED = "✓"
    SUBSTEP_COMPLETED = "🔨"

    symbol = SUBSTEP_COMPLETED if is_substep else STEP_COMPLETED
    return f"[green]{symbol}[/green] " + message


class LineBufferedOutput(io.StringIO):
    """Output stream that buffers lines and passes them to a callback."""

    LINE_REGEX = re.compile("(\r\n|\r|\n)")

    def __init__(self, callback: Callable[[str], None]):
        self._callback = callback
        self._buf = ""

    def write(self, data: str):
        chunks = self.LINE_REGEX.split(self._buf + data)

        # re.split("(<exp>)") returns the matched groups, and also the separators.
        # e.g. re.split("(+)", "a+b") returns ["a", "+", "b"].
        # This means that chunks is guaranteed to be odd in length.

        completed_lines = "".join(chunks[:-1])
        remainder = chunks[-1]

        # Partially completed lines end with a carriage return. Append a newline so that they
        # are not overwritten by the `rich.Live` and prefix the inverse operation to the remaining
        # buffer. Note that this is not perfect -- when stdout and stderr are interleaved, the results
        # can have unexpected spacing.
        if completed_lines.endswith("\r"):
            completed_lines = completed_lines[:-1] + "\n"
            # Prepend cursor up + carriage return.
            remainder = "\x1b[1A\r" + remainder

        self._callback(completed_lines)
        self._buf = remainder

    def flush(self):
        pass

    def finalize(self):
        if self._buf:
            self._callback(self._buf)
            self._buf = ""


class OutputManager:
    _visible_progress: bool
    _console: Console
    _task_states: dict[str, int]
    _task_progress_items: dict[tuple[str, int], TaskID]
    _current_render_group: Optional[Group]
    _function_progress: Optional[Progress]
    _function_queueing_progress: Optional[Progress]
    _snapshot_progress: Optional[Progress]
    _line_buffers: dict[int, LineBufferedOutput]
    _status_spinner: Spinner
    _app_page_url: Optional[str]

    def __init__(
        self, stdout: io.TextIOWrapper, show_progress: Optional[bool], status_spinner_text: str = "Running app..."
    ):
        self.stdout = stdout or sys.stdout
        if show_progress is None:
            self._visible_progress = self.stdout.isatty() or is_notebook(self.stdout)
        else:
            self._visible_progress = show_progress

        self._console = Console(file=stdout, highlight=False)
        self._task_states = {}
        self._task_progress_items = {}
        self._current_render_group = None
        self._function_progress = None
        self._function_queueing_progress = None
        self._snapshot_progress = None
        self._line_buffers = {}
        self._status_spinner = step_progress(status_spinner_text)
        self._app_page_url = None

    def print_if_visible(self, renderable) -> None:
        if self._visible_progress:
            self._console.print(renderable)

    def ctx_if_visible(self, context_mgr):
        if self._visible_progress:
            return context_mgr
        return contextlib.nullcontext()

    def make_live(self, renderable: RenderableType) -> Live:
        """Creates a customized `rich.Live` instance with the given renderable. The renderable
        is placed in a `rich.Group` to allow for dynamic additions later."""
        self._function_progress = None
        self._current_render_group = Group(renderable)
        return Live(self._current_render_group, console=self._console, transient=True, refresh_per_second=4)

    @property
    def function_progress(self) -> Progress:
        """Creates a `rich.Progress` instance with custom columns for function progress,
        and adds it to the current render group."""
        if not self._function_progress:
            self._function_progress = Progress(
                TextColumn("[progress.description][white]{task.description}[/white]"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=self._console,
            )
            if self._current_render_group:
                self._current_render_group.renderables.append(Panel(self._function_progress, style="gray50"))
        return self._function_progress

    @property
    def snapshot_progress(self) -> Progress:
        """Creates a `rich.Progress` instance with custom columns for image snapshot progress,
        and adds it to the current render group."""
        if not self._snapshot_progress:
            self._snapshot_progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TimeElapsedColumn(),
                console=self._console,
                transient=True,
            )
            if self._current_render_group:
                # Appear above function progress renderables.
                self._current_render_group.renderables.insert(0, self._snapshot_progress)
        return self._snapshot_progress

    @property
    def function_queueing_progress(self) -> Progress:
        """Creates a `rich.Progress` instance with custom columns for function queue waiting progress
        and adds it to the current render group."""
        if not self._function_queueing_progress:
            self._function_queueing_progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                FunctionQueuingColumn(),
                console=self._console,
                transient=True,
            )
            if self._current_render_group:
                self._current_render_group.renderables.append(self._function_queueing_progress)
        return self._function_queueing_progress

    def function_progress_callback(self, tag: str) -> Callable[[int, int], None]:
        """Adds a task to the current function_progress instance, and returns a callback
        to update task progress with new completed and total counts."""

        progress_task = self.function_progress.add_task(tag)

        def update_counts(completed: int, total: int):
            self.function_progress.update(progress_task, completed=completed, total=total)

        return update_counts

    def _print_log(self, fd: int, data: str) -> None:
        if fd == api_pb2.FILE_DESCRIPTOR_STDOUT:
            style = "blue"
        elif fd == api_pb2.FILE_DESCRIPTOR_STDERR:
            style = "red"
        elif fd == api_pb2.FILE_DESCRIPTOR_INFO:
            style = "yellow"
        else:
            raise Exception(f"Weird file descriptor {fd} for log output")

        self._console.out(data, style=style, end="")

    def update_app_page_url(self, app_page_url: str) -> None:
        self._app_page_url = app_page_url

    def update_task_state(self, task_id: str, state: int):
        """Updates the state of a task, sets the new task status string."""
        self._task_states[task_id] = state

        all_states = self._task_states.values()
        states_set = set(all_states)

        def tasks_at_state(state):
            return sum(x == state for x in all_states)

        # The most advanced state that's present informs the message.
        if api_pb2.TASK_STATE_ACTIVE in states_set or api_pb2.TASK_STATE_IDLE in states_set:
            tasks_running = tasks_at_state(api_pb2.TASK_STATE_ACTIVE)
            tasks_not_completed = len(self._task_states) - tasks_at_state(api_pb2.TASK_STATE_COMPLETED)
            message = f"Running ({tasks_running}/{tasks_not_completed} containers active)..."
        elif api_pb2.TASK_STATE_LOADING_IMAGE in states_set:
            tasks_loading = tasks_at_state(api_pb2.TASK_STATE_LOADING_IMAGE)
            message = f"Loading images ({tasks_loading} containers initializing)..."
        elif api_pb2.TASK_STATE_WORKER_ASSIGNED in states_set:
            message = "Worker assigned..."
        elif api_pb2.TASK_STATE_COMPLETED in states_set:
            tasks_completed = tasks_at_state(api_pb2.TASK_STATE_COMPLETED)
            message = f"Running ({tasks_completed} containers finished)..."
        else:
            message = "Running..."

        message = f"[blue]{message}[/blue] [grey70]View app at [underline]{self._app_page_url}[/underline][/grey70]"

        # Set the new message
        step_progress_update(self._status_spinner, message)

    def update_snapshot_progress(self, image_id: str, task_progress: api_pb2.TaskProgress) -> None:
        # TODO(erikbern): move this to sit on the resolver object, mostly
        completed = task_progress.pos
        total = task_progress.len

        task_key = (image_id, api_pb2.IMAGE_SNAPSHOT_UPLOAD)
        if task_key in self._task_progress_items:
            progress_task_id = self._task_progress_items[task_key]
        else:
            progress_task_id = self.snapshot_progress.add_task("[yellow]Uploading image snapshot…", total=total)
            self._task_progress_items[task_key] = progress_task_id

        try:
            self.snapshot_progress.update(progress_task_id, completed=completed, total=total)
            if completed == total:
                self.snapshot_progress.remove_task(progress_task_id)
        except KeyError:
            # Rich throws a KeyError if the task has already been removed.
            pass

    def update_queueing_progress(
        self, *, function_id: str, completed: int, total: Optional[int], description: Optional[str]
    ) -> None:
        """Handle queueing updates, ignoring completion updates for functions that have no queue progress bar."""
        task_key = (function_id, api_pb2.FUNCTION_QUEUED)
        task_description = description or f"'{function_id}' function waiting on worker"
        task_desc = f"[yellow]{task_description}. Time in queue:"
        if task_key in self._task_progress_items:
            progress_task_id = self._task_progress_items[task_key]
            try:
                self.function_queueing_progress.update(progress_task_id, completed=completed, total=total)
                if completed == total:
                    del self._task_progress_items[task_key]
                    self.function_queueing_progress.remove_task(progress_task_id)
            except KeyError:
                pass
        elif completed != total:  # Create new bar for queued function
            progress_task_id = self.function_queueing_progress.add_task(task_desc, start=True, total=None)
            self._task_progress_items[task_key] = progress_task_id

    async def put_log_content(self, log: api_pb2.TaskLogs):
        if self._visible_progress:
            stream = self._line_buffers.get(log.file_descriptor)
            if stream is None:
                stream = LineBufferedOutput(functools.partial(self._print_log, log.file_descriptor))
                self._line_buffers[log.file_descriptor] = stream
            stream.write(log.data)
        elif hasattr(self.stdout, "buffer"):
            # If we're not showing progress, there's no need to buffer lines,
            # because the progress spinner can't interfere with output.

            data = log.data.encode("utf-8")
            written = 0
            n_retries = 0
            while written < len(data):
                try:
                    written += self.stdout.buffer.write(data[written:])
                    self.stdout.flush()
                except BlockingIOError:
                    if n_retries >= 5:
                        raise
                    n_retries += 1
                    await asyncio.sleep(0.1)
        else:
            # `stdout` isn't always buffered (e.g. %%capture in Jupyter notebooks redirects it to
            # io.StringIO).
            self.stdout.write(log.data)
            self.stdout.flush()

    def flush_lines(self):
        for stream in self._line_buffers.values():
            stream.finalize()

    @contextlib.contextmanager
    def show_status_spinner(self):
        with self.ctx_if_visible(self.make_live(self._status_spinner)):
            yield


async def get_app_logs_loop(app_id: str, client: _Client, output_mgr: OutputManager):
    last_log_batch_entry_id = ""

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
            await output_mgr.put_log_content(log)

    async def _get_logs():
        nonlocal last_log_batch_entry_id

        request = api_pb2.AppGetLogsRequest(
            app_id=app_id,
            timeout=55,
            last_entry_id=last_log_batch_entry_id,
        )
        log_batch: api_pb2.TaskLogsBatch
        async for log_batch in unary_stream(client.stub.AppGetLogs, request):
            if log_batch.entry_id:
                # log_batch entry_id is empty for fd="server" messages from AppGetLogs
                last_log_batch_entry_id = log_batch.entry_id
            if log_batch.app_done:
                logger.debug("App logs are done")
                last_log_batch_entry_id = None
                break
            elif log_batch.image_id:
                # Ignore image logs - these still exist for old clients
                # New clients get them through ImageJoinStreaming (see image.py)
                pass
            else:
                for log in log_batch.items:
                    await _put_log(log_batch, log)

        output_mgr.flush_lines()

    while True:
        try:
            await _get_logs()
        except asyncio.CancelledError:
            # TODO: this should come from the backend maybe
            app_logs_url = f"https://modal.com/logs/{app_id}"
            output_mgr.print_if_visible(
                f"[red]Timed out waiting for logs. [grey70]View logs at [underline]{app_logs_url}[/underline] for remaining output.[/grey70]"
            )
            raise
        except (GRPCError, StreamTerminatedError) as exc:
            if isinstance(exc, GRPCError):
                if exc.status in RETRYABLE_GRPC_STATUS_CODES:
                    # Try again if we had a temporary connection drop,
                    # for example if computer went to sleep.
                    logger.debug("Log fetching timed out. Retrying ...")
                    continue
            elif isinstance(exc, StreamTerminatedError):
                logger.debug("Stream closed. Retrying ...")
                continue
            raise
        except Exception as exc:
            logger.exception(f"Failed to fetch logs: {exc}")
            await asyncio.sleep(1)

        if last_log_batch_entry_id is None:
            break

    logger.debug("Logging exited gracefully")
