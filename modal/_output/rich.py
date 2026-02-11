# Copyright Modal Labs 2022
"""Rich-based output management for Modal CLI.

This module contains all rich-dependent code and should only be imported when
rich output is actually needed. This allows the rest of the codebase to avoid
importing rich when output is disabled.
"""

from __future__ import annotations

import contextlib
import functools
import platform
import re
from collections.abc import Generator
from datetime import timedelta
from typing import Any, Callable

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape
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
    TransferSpeedColumn,
)
from rich.spinner import Spinner
from rich.status import Status
from rich.text import Text
from rich.tree import Tree

from modal._output.manager import (
    OutputManager,
    StatusContext,
    StatusRow,
    TransferProgressContext,
    _DisabledStatus,
    _DisabledTransferProgress,
)
from modal._utils.time_utils import timestamp_to_localized_str
from modal.config import logger
from modal.exception import DeprecationError, PendingDeprecationError, ServerWarning
from modal_proto import api_pb2

if platform.system() == "Windows":
    default_spinner = "line"
else:
    default_spinner = "dots"


def _make_console(*, stderr: bool = False, highlight: bool = True) -> Console:
    """Create a rich Console tuned for Modal CLI output.

    This is an internal function. External code should use the OutputManager
    interface (e.g., via enable_output()) instead of creating consoles directly.
    """
    return Console(
        stderr=stderr,
        highlight=highlight,
        # CLI does not work with auto-detected Jupyter HTML display_data.
        force_jupyter=False,
    )


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


class LineBufferedOutput:
    """Output stream that buffers lines and passes them to a callback."""

    LINE_REGEX = re.compile("(\r\n|\r|\n)")

    def __init__(self, callback: Callable[[str], None], show_timestamps: bool):
        self._callback = callback
        self._buf = ""
        self._show_timestamps = show_timestamps

    def write(self, log: api_pb2.TaskLogs):
        chunks = self.LINE_REGEX.split(self._buf + log.data)

        # re.split("(<exp>)") returns the matched groups, and also the separators.
        # e.g. re.split("(+)", "a+b") returns ["a", "+", "b"].
        # This means that chunks is guaranteed to be odd in length.

        if self._show_timestamps:
            for i in range(0, len(chunks) - 1, 2):
                chunks[i] = f"{timestamp_to_localized_str(log.timestamp)} {chunks[i]}"

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


class _RichStatusContext(StatusContext):
    """Wrapper around Rich's Status that implements StatusContext."""

    def __init__(self, status: Status):
        self._status = status

    def start(self) -> None:
        self._status.start()

    def stop(self) -> None:
        self._status.stop()

    def update(self, status: str) -> None:
        self._status.update(status)

    def __enter__(self) -> "_RichStatusContext":
        self._status.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._status.__exit__(*args)


class RichStatusRow(StatusRow):
    """Rich-backed implementation of StatusRow."""

    def __init__(self, progress: "Tree | None"):
        self._spinner: Spinner | None = None
        self._step_node = None
        if progress is not None:
            self._spinner = RichOutputManager._step_progress()
            self._step_node = progress.add(self._spinner)

    def message(self, message: str) -> None:
        if self._spinner is not None:
            self._spinner.update(text=message)

    def warn(self, warning: api_pb2.Warning) -> None:
        if self._step_node is not None:
            self._step_node.add(f"[yellow]:warning:[/yellow] {warning.message}")

    def finish(self, message: str) -> None:
        if self._step_node is not None and self._spinner is not None:
            self._spinner.update(text=message)
            self._step_node.label = f"🔨 {message}"


class RichOutputManager(OutputManager):
    """Rich-based implementation of OutputManager.

    Provides full terminal output with progress spinners, trees, and colored output
    using the Rich library.
    """

    _console: Console
    _stderr_console: Console
    _task_states: dict[str, int]
    _task_progress_items: dict[tuple[str, int], TaskID]
    _current_render_group: Group | None
    _function_progress: Progress | None
    _function_queueing_progress: Progress | None
    _snapshot_progress: Progress | None
    _line_buffers: dict[int, LineBufferedOutput]
    _status_spinner: Spinner | None
    _app_page_url: str | None
    __show_image_logs: bool
    _status_spinner_live: Live | None
    _object_tree: Tree | None

    @property
    def is_enabled(self) -> bool:
        return not self._quiet_mode

    @property
    def is_terminal(self) -> bool:
        return self._console.is_terminal

    @property
    def _show_image_logs(self) -> bool:
        return self.__show_image_logs

    def __init__(
        self,
        *,
        show_timestamps: bool = False,
    ):
        self._console = _make_console(highlight=False)
        self._stderr_console = _make_console(stderr=True, highlight=True)
        self._task_states = {}
        self._task_progress_items = {}
        self._current_render_group = None
        self._function_progress = None
        self._function_queueing_progress = None
        self._snapshot_progress = None
        self._line_buffers = {}
        self._status_spinner: Spinner | None = None
        self._app_page_url = None
        self.__show_image_logs = False
        self._status_spinner_live = None
        self._object_tree = None
        self._show_timestamps = show_timestamps

    @staticmethod
    def _step_progress(text: str = "") -> Spinner:
        """Returns the element to be rendered when a step is in progress (internal use)."""
        return Spinner(default_spinner, text, style="blue")

    @staticmethod
    def _step_completed_text(message: str) -> str:
        """Returns the formatted text for a step completed message (for internal use)."""
        return f"[green]✓[/green] {message}"

    def step_completed(self, message: str) -> None:
        """Print a step completed message."""
        self.print(self._step_completed_text(message))

    @contextlib.contextmanager
    def display_object_tree(self) -> Generator[None, None, None]:
        """Context manager that displays a tree of objects being created."""
        if self._quiet_mode:
            yield
            return
        self._object_tree = Tree(self._step_progress("Creating objects..."), guide_style="gray50")
        with self._make_live(self._object_tree):
            yield
        self._object_tree.label = self._step_completed_text("Created objects.")
        self.print(self._object_tree)
        self._object_tree = None

    def add_status_row(self) -> StatusRow:
        """Add a status row to the current object tree."""
        return RichStatusRow(self._object_tree)

    def print(
        self,
        renderable: Any,
        *,
        stderr: bool = False,
        highlight: bool = True,
        style: str | None = None,
        sep: str = " ",
        end: str = "\n",
    ) -> None:
        """Print a renderable to the console.

        Args:
            renderable: The content to print.
            stderr: If True, print to stderr instead of stdout.
            highlight: If True, apply syntax highlighting.
            style: Optional Rich style string (e.g., "green", "bold cyan").
            sep: The separator to use between items.
            end: The string to use at the end of the output.
        """
        if self._quiet_mode:
            return
        if stderr:
            self._stderr_console.print(renderable, highlight=highlight, style=style, sep=sep, end=end)
        else:
            self._console.print(renderable, highlight=highlight, style=style, sep=sep, end=end)

    def print_error(self, error_text: str) -> None:
        """Print an error message to stderr in a panel, ignoring quiet mode.

        This method always prints the error message regardless of quiet mode.
        """
        panel = Panel(escape(error_text), title="Error", title_align="left", border_style="red")
        self._stderr_console.print(panel, highlight=False)

    def print_json(self, data: str) -> None:
        """Print JSON data with formatting."""
        if self._quiet_mode:
            return
        self._console.print_json(data)

    def show_warning(
        self,
        warning: Warning,
        category: type[Warning],
        filename: str,
        lineno: int,
        base_showwarning: Callable[..., None],
    ) -> None:
        """Display a warning, using rich formatting for Modal-specific warnings.

        Modal warnings (DeprecationError, PendingDeprecationError, ServerWarning) are shown
        in a yellow-bordered panel with source context. Other warnings fall back to the
        default Python warning display.
        """
        # For non-Modal warnings, fall back to the default display
        if not issubclass(category, (DeprecationError, PendingDeprecationError, ServerWarning)):
            base_showwarning(warning, category, filename, lineno, file=None, line=None)
            return

        content = str(warning)
        # Extract date prefix if present (e.g., "2024-01-15 Some warning message")
        if re.match(r"^\d{4}-\d{2}-\d{2}", content):
            date = content[:10]
            message = content[11:].strip()
        else:
            date = ""
            message = content

        # Try to add source context
        try:
            with open(filename, encoding="utf-8", errors="replace") as code_file:
                source = code_file.readlines()[lineno - 1].strip()
            message = f"{message}\n\nSource: {filename}:{lineno}\n  {source}"
        except OSError:
            # e.g., when filename is "<unknown>"; raises FileNotFoundError on posix but OSError on windows
            pass

        # Build title
        if issubclass(category, ServerWarning):
            title = "Modal Warning"
        else:
            title = "Modal Deprecation Warning"
        if date:
            title += f" ({date})"

        panel = Panel(
            escape(message),
            border_style="yellow",
            title=title,
            title_align="left",
        )
        self._stderr_console.print(panel)

    def status(self, message: str) -> StatusContext:
        """Create a status spinner context manager.

        Returns a StatusContext that can be used as a context manager
        or controlled manually with start() and stop() methods.
        """
        if self._quiet_mode:
            return _DisabledStatus()
        return _RichStatusContext(self._console.status(message))

    def _make_live(self, renderable: RenderableType) -> Live:
        """Creates a customized `rich.Live` instance with the given renderable (internal use).

        The renderable is placed in a `rich.Group` to allow for dynamic additions later."""
        self._function_progress = None
        self._current_render_group = Group(renderable)
        return Live(self._current_render_group, console=self._console, transient=True, refresh_per_second=4)

    @contextlib.contextmanager
    def make_live_spinner(self, message: str) -> Generator[None, None, None]:
        """Context manager that shows a live spinner with a message."""
        spinner = self._step_progress(message)
        with self._make_live(spinner):
            yield

    def enable_image_logs(self) -> None:
        self.__show_image_logs = True

    @property
    def _function_progress_bar(self) -> Progress:
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
    def _snapshot_progress_bar(self) -> Progress:
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
    def _function_queueing_progress_bar(self) -> Progress:
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

    def function_progress_callback(self, tag: str, total: int | None) -> Callable[[int, int], None]:
        """Adds a task to the current function_progress instance, and returns a callback
        to update task progress with new completed and total counts."""

        progress_task = self._function_progress_bar.add_task(tag, total=total)

        def update_counts(completed: int, total: int):
            self._function_progress_bar.update(progress_task, completed=completed, total=total)

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

    def update_task_state(self, task_id: str, state: int) -> None:
        """Updates the state of a task, sets the new task status string."""
        self._task_states[task_id] = state

        all_states = self._task_states.values()
        states_set = set(all_states)

        def tasks_at_state(state):
            return sum(x == state for x in all_states)

        # The most advanced state that's present informs the message.
        if api_pb2.TASK_STATE_ACTIVE in states_set or api_pb2.TASK_STATE_IDLE in states_set:
            # Note that as of writing the server no longer uses TASK_STATE_ACTIVE, but we'll
            # make the numerator the sum of active / idle in case that is revived at some point in the future.
            tasks_running = tasks_at_state(api_pb2.TASK_STATE_ACTIVE) + tasks_at_state(api_pb2.TASK_STATE_IDLE)
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
        if self._status_spinner is not None:
            self._status_spinner.update(text=message)

    def update_snapshot_progress(self, image_id: str, task_progress: api_pb2.TaskProgress) -> None:
        # TODO(erikbern): move this to sit on the resolver object, mostly
        completed = task_progress.pos
        total = task_progress.len

        task_key = (image_id, api_pb2.IMAGE_SNAPSHOT_UPLOAD)
        if task_key in self._task_progress_items:
            progress_task_id = self._task_progress_items[task_key]
        else:
            progress_task_id = self._snapshot_progress_bar.add_task("[yellow]Uploading image snapshot…", total=total)
            self._task_progress_items[task_key] = progress_task_id

        try:
            self._snapshot_progress_bar.update(progress_task_id, completed=completed, total=total)
            if completed == total:
                self._snapshot_progress_bar.remove_task(progress_task_id)
        except KeyError:
            # Rich throws a KeyError if the task has already been removed.
            pass

    def update_queueing_progress(
        self, *, function_id: str, completed: int, total: int | None, description: str | None
    ) -> None:
        """Handle queueing updates, ignoring completion updates for functions that have no queue progress bar."""
        task_key = (function_id, api_pb2.FUNCTION_QUEUED)
        task_description = description or f"'{function_id}' function waiting on worker"
        task_desc = f"[yellow]{task_description}. Time in queue:"
        if task_key in self._task_progress_items:
            progress_task_id = self._task_progress_items[task_key]
            try:
                self._function_queueing_progress_bar.update(progress_task_id, completed=completed, total=total)
                if completed == total:
                    del self._task_progress_items[task_key]
                    self._function_queueing_progress_bar.remove_task(progress_task_id)
            except KeyError:
                pass
        elif completed != total:  # Create new bar for queued function
            progress_task_id = self._function_queueing_progress_bar.add_task(task_desc, start=True, total=None)
            self._task_progress_items[task_key] = progress_task_id

    async def put_log_content(self, log: api_pb2.TaskLogs) -> None:
        """Process and display log content.

        Note: Log output is always displayed even when quiet mode is enabled.
        Quiet mode suppresses progress indicators and status messages, but not
        actual log output from running functions/images.
        """
        stream = self._line_buffers.get(log.file_descriptor)
        if stream is None:
            stream = LineBufferedOutput(functools.partial(self._print_log, log.file_descriptor), self._show_timestamps)
            self._line_buffers[log.file_descriptor] = stream
        stream.write(log)

    def flush_lines(self) -> None:
        for stream in self._line_buffers.values():
            stream.finalize()

    @contextlib.contextmanager
    def transfer_progress(self, type: str) -> Generator[TransferProgressContext, None, None]:
        """Context manager for tracking file transfer progress.

        Args:
            type: Either "download" or "upload".

        Yields:
            A TransferProgressContext with a progress() method for updating transfer progress.
        """
        if self._quiet_mode:
            yield _DisabledTransferProgress()
            return
        handler = ProgressHandler(type, self._console)
        with handler.live:
            yield _RichTransferProgress(handler)

    @contextlib.contextmanager
    def show_status_spinner(self, status_text: str = "Running app...") -> Generator[None, None, None]:
        if self._quiet_mode:
            yield
            return
        self._status_spinner = RichOutputManager._step_progress(status_text)
        self._status_spinner_live = self._make_live(self._status_spinner)
        with self._status_spinner_live:
            yield

    def stop_status_spinner(self) -> None:
        """Stop the status spinner if it's running."""
        if self._status_spinner_live:
            self._status_spinner_live.stop()
            self._status_spinner_live = None


class _RichTransferProgress(TransferProgressContext):
    """Rich-backed transfer progress context.

    Wraps a ProgressHandler to provide the TransferProgressContext interface.
    """

    def __init__(self, handler: "ProgressHandler"):
        self._handler = handler

    def progress(
        self,
        task_id: TaskID | None = None,
        advance: float | None = None,
        name: str | None = None,
        size: float | None = None,
        reset: bool | None = False,
        complete: bool | None = False,
    ) -> TaskID | None:
        return self._handler.progress(
            task_id=task_id,
            advance=advance,
            name=name,
            size=size,
            reset=reset,
            complete=complete,
        )


class ProgressHandler:
    """Internal handler for rich-based transfer progress display.

    This class is used internally by RichOutputManager.transfer_progress().
    """

    live: Live
    _type: str
    _spinner: Spinner
    _overall_progress: Progress
    _download_progress: Progress
    _overall_progress_task_id: TaskID
    _total_tasks: int
    _completed_tasks: int

    def __init__(self, type: str, console: Console):
        self._type = type

        if self._type == "download":
            title = "Downloading file(s) to local..."
        elif self._type == "upload":
            title = "Uploading file(s) to volume..."
        else:
            raise NotImplementedError(f"Progress handler of type: `{type}` not yet implemented")

        self._spinner = RichOutputManager._step_progress(title)

        self._overall_progress = Progress(
            TextColumn(f"[bold white]{title}", justify="right"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TextColumn("[bold white]{task.description}"),
            transient=True,
            console=console,
        )
        self._download_progress = Progress(
            TextColumn("[bold white]{task.fields[path]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            transient=True,
            console=console,
        )

        self.live = Live(
            Group(self._spinner, self._overall_progress, self._download_progress), transient=True, refresh_per_second=4
        )

        self._overall_progress_task_id = self._overall_progress.add_task(".", start=True)
        self._total_tasks = 0
        self._completed_tasks = 0

    def _add_sub_task(self, name: str, size: float) -> TaskID:
        task_id = self._download_progress.add_task(self._type, path=name, start=True, total=size)
        self._total_tasks += 1
        self._overall_progress.update(self._overall_progress_task_id, total=self._total_tasks)
        return task_id

    def _reset_sub_task(self, task_id: TaskID):
        self._download_progress.reset(task_id)

    def _complete_progress(self):
        # TODO: we could probably implement some callback progression from the server
        # to get progress reports for the post processing too
        # so we don't have to just spin here
        self._overall_progress.remove_task(self._overall_progress_task_id)
        self._spinner.update(text="Post processing...")

    def _complete_sub_task(self, task_id: TaskID):
        self._completed_tasks += 1
        self._download_progress.remove_task(task_id)
        self._overall_progress.update(
            self._overall_progress_task_id,
            advance=1,
            description=f"({self._completed_tasks} out of {self._total_tasks} files completed)",
        )

    def _advance_sub_task(self, task_id: TaskID, advance: float):
        self._download_progress.update(task_id, advance=advance)

    def progress(
        self,
        task_id: TaskID | None = None,
        advance: float | None = None,
        name: str | None = None,
        size: float | None = None,
        reset: bool | None = False,
        complete: bool | None = False,
    ) -> TaskID | None:
        try:
            if task_id is not None:
                if reset:
                    return self._reset_sub_task(task_id)
                elif complete:
                    return self._complete_sub_task(task_id)
                elif advance is not None:
                    return self._advance_sub_task(task_id, advance)
            elif name is not None and size is not None:
                return self._add_sub_task(name, size)
            elif complete:
                return self._complete_progress()
        except Exception as exc:
            # Liberal exception handling to avoid crashing downloads and uploads.
            logger.error(f"failed progress update: {exc}")
        raise NotImplementedError(
            "Unknown action to take with args: "
            + f"name={name} "
            + f"size={size} "
            + f"task_id={task_id} "
            + f"advance={advance} "
            + f"reset={reset} "
            + f"complete={complete} "
        )
