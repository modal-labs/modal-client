# Copyright Modal Labs 2025
"""Tests for the output management system."""

from io import StringIO

from rich.console import Console

from modal._output.manager import DisabledOutputManager
from modal._output.rich import RichOutputManager


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


def test_can_process_logs_distinguishes_disabled_from_quiet():
    """Regression test for issue #2076.

    `modal run -q` is documented as "Don't show Modal progress indicators" — but historically
    it also suppressed all stdout/stderr from remote functions. The root cause was that
    `runner._run_app` gated starting the app log-streaming loop on `output_mgr.is_enabled`,
    which returns False both when output is fully disabled (no `enable_output()`) and when
    quiet mode is active. As a result, `--quiet` skipped starting the logs loop entirely.

    `can_process_logs` is the new signal callers should use to decide whether logs should be
    streamed: True for any real (Rich) manager — even in quiet mode — and False only when the
    manager is the no-op `DisabledOutputManager`. Quiet mode keeps suppressing progress UI
    via `is_enabled`, but no longer suppresses log delivery.
    """
    # DisabledOutputManager: no logs (no enable_output() context).
    disabled = DisabledOutputManager()
    assert disabled.can_process_logs is False
    assert disabled.is_enabled is False

    # RichOutputManager, quiet mode off: spinners on, logs on.
    rich_mgr = RichOutputManager()
    rich_mgr.set_quiet_mode(False)
    assert rich_mgr.can_process_logs is True
    assert rich_mgr.is_enabled is True

    # RichOutputManager, quiet mode on: spinners off, but logs MUST still flow.
    rich_mgr.set_quiet_mode(True)
    assert rich_mgr.can_process_logs is True, "Quiet mode must NOT prevent log streaming — only progress indicators."
    assert rich_mgr.is_enabled is False


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
