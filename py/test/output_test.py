# Copyright Modal Labs 2025
"""Tests for the output management system."""

import time
from io import StringIO

from rich.console import Console

from modal._output.rich import QUEUEING_STRAGGLER_GRACE_SECS, RichOutputManager


def test_print_suppressed_in_quiet_mode():
    """Verify that print() is suppressed when quiet mode is enabled."""
    # Create a RichOutputManager with a custom console that captures output
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    manager = RichOutputManager()
    manager._console = console

    # Without quiet mode, print() should produce output
    manager.set_quiet_mode(False)
    manager.print("Test message")

    output_before_quiet = output.getvalue()
    assert "Test message" in output_before_quiet, "Expected output when quiet mode is disabled"

    # Record the current output length
    output_length_before = len(output_before_quiet)

    # With quiet mode enabled, print() should not produce any additional output
    manager.set_quiet_mode(True)
    manager.print("Another message")

    output_after_quiet = output.getvalue()
    assert len(output_after_quiet) == output_length_before, "Expected no additional output when quiet mode is enabled"
    assert "Another message" not in output_after_quiet, "Quiet mode message should not appear in output"


def test_status_suppressed_in_quiet_mode(monkeypatch):
    """Verify that status() is suppressed when quiet mode is enabled.

    We verify that:
    1. In non-quiet mode, status produces output to the console
    2. In quiet mode, status produces no output
    """
    # Rich's Live display requires a non-dumb terminal to render output during the context.
    # Set TERM to enable full terminal output capture.
    monkeypatch.setenv("TERM", "xterm-256color")

    # Create a RichOutputManager with a custom console that captures output
    output = StringIO()
    console = Console(file=output, force_terminal=True, force_interactive=True, width=80)
    manager = RichOutputManager()
    manager._console = console

    # Without quiet mode, status should produce output
    manager.set_quiet_mode(False)
    with manager.status("Test status") as status_ctx:
        status_ctx.update("Updated status")

    output_before_quiet = output.getvalue()
    assert len(output_before_quiet) > 0, "Expected output when status is used without quiet mode"
    assert "Updated status" in output_before_quiet, "Expected status message in output"

    # Record the current output length
    output_length_before = len(output_before_quiet)

    # With quiet mode enabled, status should not produce any additional output
    manager.set_quiet_mode(True)
    with manager.status("Quiet status") as quiet_status_ctx:
        quiet_status_ctx.update("This should be silently ignored")

    output_after_quiet = output.getvalue()
    assert len(output_after_quiet) == output_length_before, "Expected no additional output when quiet mode is enabled"


def _queued_update(manager: RichOutputManager, function_id: str, seconds_in_queue: int) -> None:
    manager.update_queueing_progress(
        function_id=function_id,
        completed=seconds_in_queue,
        total=0,
        description=f"Function {function_id} is waiting to be scheduled",
    )


def _scheduled_update(manager: RichOutputManager, function_id: str) -> None:
    manager.update_queueing_progress(function_id=function_id, completed=1, total=1, description=None)


def _queueing_bar_task_count(manager: RichOutputManager) -> int:
    progress = manager._function_queueing_progress
    return len(progress.tasks) if progress else 0


def test_queueing_progress_ignores_straggler_update():
    """A queueing update published just before the function was scheduled must not revive the bar."""
    manager = RichOutputManager()
    _queued_update(manager, "fu-123", 11)
    assert _queueing_bar_task_count(manager) == 1

    _scheduled_update(manager, "fu-123")

    _queued_update(manager, "fu-123", 12)
    assert _queueing_bar_task_count(manager) == 0


def test_queueing_progress_returns_for_later_queueing(monkeypatch):
    """Once the grace period passes, a function that queues again gets its bar back."""
    manager = RichOutputManager()
    _queued_update(manager, "fu-123", 11)
    _scheduled_update(manager, "fu-123")

    now = time.monotonic() + QUEUEING_STRAGGLER_GRACE_SECS + 1
    monkeypatch.setattr(time, "monotonic", lambda: now)
    _queued_update(manager, "fu-123", 11)
    assert _queueing_bar_task_count(manager) == 1
