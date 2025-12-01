# Copyright Modal Labs 2025
import contextlib
import os
import pty
import sys

import modal
from modal.runner import interactive_shell


@contextlib.contextmanager
def pty_stdin():
    parent_pty, child_pty = pty.openpty()
    try:
        with os.fdopen(child_pty, "rb", buffering=0) as f:
            yield f
    finally:
        os.close(parent_pty)


def test_shell_handles_non_unicode(monkeypatch):
    """Test that the modal shell handles non-unicode bytes.

    Many terminal applications (notably vi) send non-unicode bytes to stdout.
    """
    app = modal.App("sdk-test-app")

    # Create a pseudo-terminal to provide a real TTY for the test.
    # The attach() method called from _interactive_shell() requires sys.stdin to be a real terminal.
    with pty_stdin() as f:
        monkeypatch.setattr(sys, "stdin", f)

        # Note: with pty=True, output goes to the PTY which gets printed during attach(),
        # so the test just verifies that non-unicode bytes don't cause a crash
        interactive_shell(app, ["python", "-c", "import sys; sys.stdout.buffer.write(b'hello\\xbdworld')"], pty=True)
