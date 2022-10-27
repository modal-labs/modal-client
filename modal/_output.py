# Copyright Modal Labs 2022
import asyncio
import contextlib
import functools
import io
import platform
import re
import sys
from typing import Callable, Dict, Optional

from grpclib.exceptions import GRPCError, StreamTerminatedError
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
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


def step_progress(text: str = "") -> Spinner:
    """Returns the element to be rendered when a step is in progress."""
    return Spinner(default_spinner, text, style="blue")


def step_progress_update(spinner: Spinner, message: str):
    spinner.update(text=Text(message, style="blue"))


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
        for i in range(int(len(chunks) / 2)):
            # piece together chunk back with separator.
            line = chunks[2 * i] + chunks[2 * i + 1]
            self._callback(line)

        self._buf = chunks[-1]

    def flush(self):
        pass

    def finalize(self):
        if self._buf:
            self._callback(self._buf)
            self._buf = ""


class OutputManager:
    _visible_progress: bool
    _console: Console
    _task_states: Dict[str, int]

    def __init__(self, stdout, show_progress: Optional[bool]):
        self.stdout = stdout or sys.stdout
        if show_progress is None:
            self._visible_progress = self.stdout.isatty() or is_notebook(self.stdout)
        else:
            self._visible_progress = show_progress

        self._console = Console(file=stdout, highlight=False)
        self._task_states = {}
        self._task_progress_items: dict[tuple[str, api_pb2.ProgressType], TaskID] = {}
        self._current_render_group: Optional[Group] = None
        self._function_progress: Optional[Progress] = None
        self._task_progress: Optional[Progress] = None

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
        return Live(self._current_render_group, console=self._console, transient=True, refresh_per_second=10)

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
    def task_progress(self) -> Progress:
        """Creates a `rich.Progress` instance with custom columns for task progress,
        and adds it to the current render group."""
        if not self._task_progress:
            self._task_progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TimeElapsedColumn(),
                console=self._console,
                transient=True,
            )
            if self._current_render_group:
                # Appear above function progress renderables.
                self._current_render_group.renderables.insert(0, self._task_progress)
        return self._task_progress

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

    def _update_task_state(self, task_id: str, state: int) -> str:
        """Updates the state of a task, returning the new task status string."""
        self._task_states[task_id] = state

        all_states = self._task_states.values()
        states_set = set(all_states)

        def tasks_at_state(state):
            return sum(x == state for x in all_states)

        # The most advanced state that's present informs the message.
        if api_pb2.TASK_STATE_ACTIVE in states_set or api_pb2.TASK_STATE_IDLE in states_set:
            tasks_running = tasks_at_state(api_pb2.TASK_STATE_ACTIVE)
            tasks_not_completed = len(self._task_states) - tasks_at_state(api_pb2.TASK_STATE_COMPLETED)
            return f"Running ({tasks_running}/{tasks_not_completed} containers active)..."
        elif api_pb2.TASK_STATE_LOADING_IMAGE in states_set:
            tasks_loading = tasks_at_state(api_pb2.TASK_STATE_LOADING_IMAGE)
            return f"Loading images ({tasks_loading} containers initializing)..."
        elif api_pb2.TASK_STATE_WORKER_ASSIGNED in states_set:
            return "Worker assigned..."
        elif api_pb2.TASK_STATE_COMPLETED in states_set:
            tasks_completed = tasks_at_state(api_pb2.TASK_STATE_COMPLETED)
            return f"Running ({tasks_completed} containers finished)..."
        else:
            return "Running..."

    def _update_task_progress(self, *, task_id: str, progress_task, completed: int, total: int) -> None:
        key = (task_id, progress_task)

        if progress_task == api_pb2.IMAGE_SNAPSHOT_UPLOAD:
            progress_task_display = "Uploading image snapshot…"
        else:
            raise Exception(f"Unknown {progress_task} type for progress update.")

        task_desc = f"[yellow]{progress_task_display}"

        if key in self._task_progress_items:
            progress_task_id = self._task_progress_items[key]
        else:
            progress_task_id = self.task_progress.add_task(task_desc, total=total)
            self._task_progress_items[key] = progress_task_id

        try:
            self.task_progress.update(progress_task_id, completed=completed, total=total)
            if completed == total:
                self.task_progress.remove_task(progress_task_id)
        except KeyError:
            # Rich throws a KeyError if the task has already been removed.
            pass

    async def get_logs_loop(self, app_id: str, client: _Client, status_spinner: Spinner, last_log_batch_entry_id: str):
        async def _get_logs():
            nonlocal last_log_batch_entry_id

            request = api_pb2.AppGetLogsRequest(
                app_id=app_id,
                timeout=55,
                last_entry_id=last_log_batch_entry_id,
            )
            log_batch: api_pb2.TaskLogsBatch
            line_buffers: Dict[int, LineBufferedOutput] = {}
            async for log_batch in unary_stream(client.stub.AppGetLogs, request):
                if log_batch.app_done:
                    logger.debug("App logs are done")
                    last_log_batch_entry_id = None
                    return
                else:
                    if log_batch.entry_id != "":
                        # log_batch entry_id is empty for fd="server" messages from AppGetLogs
                        last_log_batch_entry_id = log_batch.entry_id

                    for log in log_batch.items:
                        if log.task_state:
                            message = self._update_task_state(log_batch.task_id, log.task_state)
                            step_progress_update(status_spinner, message)
                        if log.task_progress.len or log.task_progress.pos:
                            self._update_task_progress(
                                task_id=log_batch.task_id,
                                progress_task=log.task_progress.progress_type,
                                completed=log.task_progress.pos or 0,
                                total=log.task_progress.len or 0,
                            )
                        elif log.data:
                            if self._visible_progress:
                                stream = line_buffers.get(log.file_descriptor)
                                if stream is None:
                                    stream = LineBufferedOutput(functools.partial(self._print_log, log.file_descriptor))
                                    line_buffers[log.file_descriptor] = stream
                                stream.write(log.data)
                            else:
                                # If we're not showing progress, there's no need to buffer lines,
                                # because the progress spinner can't interfere with output.
                                self.stdout.write(log.data)
                                self.stdout.flush()
            for stream in line_buffers.values():
                stream.finalize()

        while True:
            try:
                await _get_logs()
            except asyncio.CancelledError:
                logger.debug("Logging cancelled")
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

            if last_log_batch_entry_id is None:
                break
            # TODO: catch errors, sleep, and retry?
        logger.debug("Logging exited gracefully")
