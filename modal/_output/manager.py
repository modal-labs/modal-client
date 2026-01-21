# Copyright Modal Labs 2022
"""Output management interface for Modal CLI.

This module defines the interface for output management and provides a no-op
implementation for when output is disabled. The rich-based implementation lives
in rich.py to avoid importing rich when it's not needed.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Generator
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modal_proto import api_pb2


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
