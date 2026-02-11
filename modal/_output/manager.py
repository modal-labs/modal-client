# Copyright Modal Labs 2022
"""Output management interface for Modal CLI.

This module defines the interface for output management and provides a no-op
implementation for when output is disabled. The rich-based implementation lives
in rich.py to avoid importing rich when it's not needed.
"""

import asyncio
import contextlib
import sys
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Optional

from modal._utils.shell_utils import write_to_fd

if TYPE_CHECKING:
    from modal_proto import api_pb2


class StatusContext(ABC):
    """Abstract base class for status context managers that support manual start/stop control."""

    @abstractmethod
    def start(self) -> None:
        """Start showing the status spinner."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop showing the status spinner."""
        ...

    @abstractmethod
    def update(self, status: str) -> None:
        """Update the status message."""
        ...

    @abstractmethod
    def __enter__(self) -> "StatusContext": ...

    @abstractmethod
    def __exit__(self, *args: Any) -> None: ...


class StatusRow(ABC):
    """Abstract base class describing a row in the object creation status tree."""

    @abstractmethod
    def message(self, message: str) -> None:
        """Update the primary message shown for this row."""
        ...

    @abstractmethod
    def warn(self, warning: "api_pb2.Warning") -> None:
        """Append a warning message associated with this row."""
        ...

    @abstractmethod
    def finish(self, message: str) -> None:
        """Mark the row as finished with the given message."""
        ...


class DisabledStatusRow(StatusRow):
    """No-op StatusRow used when output is disabled."""

    def message(self, message: str) -> None:
        pass

    def warn(self, warning: "api_pb2.Warning") -> None:
        pass

    def finish(self, message: str) -> None:
        pass


class TransferProgressContext(ABC):
    """Abstract base class for transfer progress tracking context."""

    @abstractmethod
    def progress(
        self,
        task_id: Any = None,
        advance: Optional[float] = None,
        name: Optional[str] = None,
        size: Optional[float] = None,
        reset: Optional[bool] = False,
        complete: Optional[bool] = False,
    ) -> Any:
        """Update progress. Returns task_id when creating a new task."""
        ...


class _DisabledTransferProgress(TransferProgressContext):
    """No-op transfer progress context for when output is disabled."""

    def progress(
        self,
        task_id: Any = None,
        advance: Optional[float] = None,
        name: Optional[str] = None,
        size: Optional[float] = None,
        reset: Optional[bool] = False,
        complete: Optional[bool] = False,
    ) -> None:
        return None


class OutputManager(ABC):
    """Abstract base class defining the interface for output management.

    This class allows for different implementations:
    - RichOutputManager: Full rich-based terminal output with progress spinners, trees, etc.
    - DisabledOutputManager: No-op implementation for when output is disabled.

    Subclasses must implement all abstract methods. Use OutputManager.get() to get
    the current output manager instance.
    """

    # Settings that can be modified at runtime - shared by all implementations
    _quiet_mode: bool = False
    _show_timestamps: bool = False

    @classmethod
    def get(cls) -> "OutputManager":
        """Get the current output manager.

        Returns a RichOutputManager when output is enabled, otherwise returns
        a DisabledOutputManager that provides no-op implementations of all methods.

        This allows code to call output methods without checking if output is enabled,
        simplifying the calling code.
        """
        return _current_output_manager

    @classmethod
    def _set(cls, manager: "OutputManager") -> None:
        """Set the current output manager. Used internally by enable_output()."""
        global _current_output_manager
        _current_output_manager = manager

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether output is currently being displayed somewhere.

        Returns False if using DisabledOutputManager OR if quiet mode is active.
        This reflects whether print(), status(), etc. will actually produce visible output.
        """
        ...

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """Whether the output is connected to a terminal (TTY)."""
        ...

    @property
    @abstractmethod
    def _show_image_logs(self) -> bool:
        """Whether to show image logs."""
        ...

    @abstractmethod
    @contextlib.contextmanager
    def display_object_tree(self) -> Generator[None, None, None]:
        """Context manager that displays a tree of objects being created."""
        ...

    @abstractmethod
    def add_status_row(self) -> StatusRow:
        """Add a status row to the current object tree."""
        ...

    @abstractmethod
    def print(
        self,
        renderable: Any,
        *,
        stderr: bool = False,
        highlight: bool = True,
        style: Optional[str] = None,
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
        ...

    @abstractmethod
    def print_error(self, error_text: str) -> None:
        """Print an error message to stderr, ignoring quiet/disabled mode.

        This method always prints the error message regardless of quiet mode
        or whether output is disabled. It should be used for critical error
        messages that must always be visible to the user.

        Args:
            error_text: The error message text to display.
        """
        ...

    @abstractmethod
    def show_warning(
        self,
        warning: Warning,
        category: type[Warning],
        filename: str,
        lineno: int,
        base_showwarning: Callable[..., None],
    ) -> None:
        """Display a warning message.

        This method is called by the patched warnings.showwarning for all warnings.

        Args:
            warning: The warning instance.
            category: The warning category (class).
            filename: The source file where the warning originated.
            lineno: The line number where the warning originated.
            base_showwarning: The original warnings.showwarning function to fall back to.
        """
        ...

    @abstractmethod
    def print_json(self, data: str) -> None:
        """Print JSON data with formatting."""
        ...

    @abstractmethod
    def status(self, message: str) -> "StatusContext":
        """Context manager that displays a status spinner with a message.

        Returns a context manager that shows a spinner while active.
        The returned object has start(), stop(), and update() methods for manual control.
        """
        ...

    @abstractmethod
    def make_live_spinner(self, message: str) -> AbstractContextManager[None]:
        """Context manager that shows a live spinner with a message.

        This combines spinner creation and live display into a single interface,
        hiding the implementation details of how spinners are rendered.
        """
        ...

    @abstractmethod
    def enable_image_logs(self) -> None:
        """Enable showing image logs."""
        ...

    @abstractmethod
    def show_status_spinner(self, status_text: str = "Running app...") -> AbstractContextManager[None]:
        """Context manager that shows a status spinner.

        Args:
            status_text: The text to display next to the spinner. Defaults to "Running app...".
        """
        ...

    @abstractmethod
    def update_app_page_url(self, app_page_url: str) -> None:
        """Update the app page URL for display."""
        ...

    @abstractmethod
    def function_progress_callback(self, tag: str, total: Optional[int]) -> Callable[[int, int], None]:
        """Get a callback for updating function progress."""
        ...

    @abstractmethod
    def update_task_state(self, task_id: str, state: int) -> None:
        """Update the state of a task."""
        ...

    @abstractmethod
    def update_snapshot_progress(self, image_id: str, task_progress: "api_pb2.TaskProgress") -> None:
        """Update snapshot upload progress."""
        ...

    @abstractmethod
    def update_queueing_progress(
        self, *, function_id: str, completed: int, total: Optional[int], description: Optional[str]
    ) -> None:
        """Update function queueing progress."""
        ...

    @abstractmethod
    async def put_log_content(self, log: "api_pb2.TaskLogs") -> None:
        """Process and display log content.

        Note: In RichOutputManager, log output is always displayed even when quiet mode
        is enabled. This is intentional - quiet mode suppresses progress indicators and
        status messages, but not actual log output from running functions/images.
        In contrast, DisabledOutputManager suppresses all output including logs.
        """
        ...

    async def put_pty_content(self, log: "api_pb2.TaskLogs") -> None:
        """Write PTY content to stdout for interactive terminal sessions.

        This handles raw PTY output from interactive shells/debuggers, writing
        directly to stdout with proper handling of non-blocking writes.

        Note: This method is implemented in the base class (not abstract) because
        interactive/raw terminal content is always forwarded to stdout regardless
        of whether output is enabled or quiet mode is active. This ensures that
        interactive debugging sessions (e.g., pdb breakpoints) work correctly.
        """
        stdout = sys.stdout
        if hasattr(stdout, "buffer"):
            data = log.data.encode("utf-8")
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
            # stdout isn't always buffered (e.g. %%capture in Jupyter notebooks redirects it to io.StringIO)
            stdout.write(log.data)
            stdout.flush()

    @abstractmethod
    def flush_lines(self) -> None:
        """Flush any buffered output."""
        ...

    @abstractmethod
    def transfer_progress(self, type: str) -> AbstractContextManager["TransferProgressContext"]:
        """Context manager for tracking file transfer progress.

        Args:
            type: Either "download" or "upload".

        Returns:
            A context manager that yields a TransferProgressContext with a progress() method.
        """
        ...

    @abstractmethod
    def stop_status_spinner(self) -> None:
        """Stop the status spinner if it's running.

        This is used to cleanly stop the spinner before entering PTY mode or
        other situations where the spinner would interfere with output.
        """
        ...

    def set_quiet_mode(self, quiet: bool) -> None:
        """Enable or disable quiet mode.

        When quiet mode is enabled:
        - All print() output is suppressed, including stderr
        - Progress indicators (spinners, progress bars) are suppressed
        - Warnings (via show_warning) are still displayed
        - Log output (via put_log_content) is still displayed
        - Interactive/raw PTY content (via put_pty_content) is still forwarded to stdout
        - Error output (via print_error) is still displayed

        Note: This differs from DisabledOutputManager, which suppresses all output
        including logs. Quiet mode is meant for reducing noise while still showing
        relevant content.
        """
        self._quiet_mode = quiet

    def set_timestamps(self, show: bool) -> None:
        """Enable or disable timestamp display in log output."""
        self._show_timestamps = show

    @abstractmethod
    def step_completed(self, message: str) -> None:
        """Print a step completed message with appropriate formatting."""
        ...


class _DisabledStatus(StatusContext):
    """No-op status context manager for when output is disabled."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def update(self, status: str) -> None:
        pass

    def __enter__(self) -> "_DisabledStatus":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class DisabledOutputManager(OutputManager):
    """No-op implementation of OutputManager for when output is disabled.

    Most methods are no-ops that do nothing, allowing code to call output methods
    without checking if the output manager exists.

    Exceptions:
    - Interactive/raw PTY content (via put_pty_content) is always forwarded to stdout
      to ensure interactive debugging sessions work correctly.
    - Warnings (via show_warning) fall back to Python's default warning display.
    """

    @property
    def is_enabled(self) -> bool:
        return False

    @property
    def is_terminal(self) -> bool:
        return sys.stdout.isatty()

    @property
    def _show_image_logs(self) -> bool:
        return True  # Always "show" logs when disabled (don't filter them)

    @contextlib.contextmanager
    def display_object_tree(self) -> Generator[None, None, None]:
        yield

    def add_status_row(self) -> StatusRow:
        return DisabledStatusRow()

    def print(
        self,
        renderable: Any,
        *,
        stderr: bool = False,
        highlight: bool = True,
        style: Optional[str] = None,
        sep: str = " ",
        end: str = "\n",
    ) -> None:
        pass

    def print_error(self, error_text: str) -> None:
        # Print error to stderr without any formatting (Rich is not available)
        sys.stderr.write(error_text + "\n")

    def show_warning(
        self,
        warning: Warning,
        category: type[Warning],
        filename: str,
        lineno: int,
        base_showwarning: Callable[..., None],
    ) -> None:
        # When output is disabled, fall back to the default warning display
        base_showwarning(warning, category, filename, lineno, file=None, line=None)

    def print_json(self, data: str) -> None:
        pass

    def status(self, message: str) -> StatusContext:
        return _DisabledStatus()

    @contextlib.contextmanager
    def make_live_spinner(self, message: str) -> Generator[None, None, None]:
        yield

    def enable_image_logs(self) -> None:
        pass

    @contextlib.contextmanager
    def show_status_spinner(self, status_text: str = "Running app...") -> Generator[None, None, None]:
        yield

    def update_app_page_url(self, app_page_url: str) -> None:
        pass

    def function_progress_callback(self, tag: str, total: Optional[int]) -> Callable[[int, int], None]:
        def noop(completed: int, total: int) -> None:
            pass

        return noop

    def update_task_state(self, task_id: str, state: int) -> None:
        pass

    def update_snapshot_progress(self, image_id: str, task_progress: "api_pb2.TaskProgress") -> None:
        pass

    def update_queueing_progress(
        self, *, function_id: str, completed: int, total: Optional[int], description: Optional[str]
    ) -> None:
        pass

    async def put_log_content(self, log: "api_pb2.TaskLogs") -> None:
        pass

    def flush_lines(self) -> None:
        pass

    @contextlib.contextmanager
    def transfer_progress(self, type: str) -> Generator[TransferProgressContext, None, None]:
        """No-op transfer progress context manager."""
        yield _DisabledTransferProgress()

    def stop_status_spinner(self) -> None:
        pass

    def step_completed(self, message: str) -> None:
        pass


# Singleton instance of the disabled output manager
_DISABLED_OUTPUT_MANAGER = DisabledOutputManager()


# Module-level state for output management
_current_output_manager: OutputManager = _DISABLED_OUTPUT_MANAGER
