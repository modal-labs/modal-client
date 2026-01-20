# Copyright Modal Labs 2024
"""Interface to Modal's OutputManager functionality.

These functions live here so that Modal library code can import them without
transitively importing Rich, as we do in _rich_output.py. This allows
us to avoid importing Rich for client code that runs in the container environment.
"""

import contextlib
from collections.abc import Generator

from ._output import _DISABLED_OUTPUT_MANAGER, OutputManager

# Module-level state for output management
_current_output_manager: OutputManager = _DISABLED_OUTPUT_MANAGER


@contextlib.contextmanager
def enable_output(show_progress: bool = True, show_timestamps: bool = False) -> Generator[OutputManager, None, None]:
    """Context manager that enable output when using the Python SDK.

    This will print to stdout and stderr things such as
    1. Logs from running functions
    2. Status of creating objects
    3. Map progress

    Example:
    ```python
    app = modal.App()
    with modal.enable_output():
        with app.run():
            ...
    ```
    """
    global _current_output_manager
    previous_output_manager = _current_output_manager

    if show_progress:
        from ._rich_output import RichOutputManager

        _current_output_manager = RichOutputManager(show_timestamps=show_timestamps)
    try:
        yield _current_output_manager
    finally:
        _current_output_manager = previous_output_manager


def _get_output_manager() -> OutputManager:
    """Get the current output manager.

    Returns a RichOutputManager when output is enabled, otherwise returns
    a DisabledOutputManager that provides no-op implementations of all methods.

    This allows code to call output methods without checking if output is enabled,
    simplifying the calling code.
    """
    return _current_output_manager


def _disable_output_manager() -> None:
    """Disable the current output manager.

    This is called by RichOutputManager.disable() to ensure that subsequent calls
    to _get_output_manager() return a DisabledOutputManager.
    """
    global _current_output_manager
    _current_output_manager = _DISABLED_OUTPUT_MANAGER
