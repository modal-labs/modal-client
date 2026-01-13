# Copyright Modal Labs 2024
"""Interface to Modal's OutputManager functionality.

These functions live here so that Modal library code can import them without
transitively importing Rich, as we do in global scope in _output.py. This allows
us to avoid importing Rich for client code that runs in the container environment.

"""

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ._output import DisabledOutputManager, RichOutputManager


OUTPUT_ENABLED = False


@contextlib.contextmanager
def enable_output(
    show_progress: bool = True, show_timestamps: bool = False
) -> Generator["RichOutputManager | None", None, None]:
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
    from ._output import RichOutputManager

    # Toggle the output flag from within this function so that we can
    # call _get_output_manager from within the library and only import
    # the _output module if output is explicitly enabled. That prevents
    # us from trying to import rich inside a container environment where
    # it might not be installed. This is sort of hacky and I would prefer
    # a more thorough refactor where the OutputManager is fully useable
    # without rich installed, but that's a larger project.
    global OUTPUT_ENABLED

    try:
        with RichOutputManager.enable_output(show_progress, show_timestamps=show_timestamps) as mgr:
            OUTPUT_ENABLED = True
            yield mgr
    finally:
        OUTPUT_ENABLED = False


def _get_output_manager() -> "Union[RichOutputManager, DisabledOutputManager]":
    """Get the current output manager.

    Returns a RichOutputManager when output is enabled, otherwise returns
    a DisabledOutputManager that provides no-op implementations of all methods.

    This allows code to call output methods without checking if output is enabled,
    simplifying the calling code.
    """
    if OUTPUT_ENABLED:
        from ._output import RichOutputManager

        mgr = RichOutputManager.get()
        if mgr is not None:
            return mgr

    # Return the singleton disabled output manager
    from ._output import _DISABLED_OUTPUT_MANAGER

    return _DISABLED_OUTPUT_MANAGER


def _is_output_enabled() -> bool:
    """Check if rich output is enabled.

    This is useful for code that needs to conditionally perform operations
    based on whether output is truly enabled (e.g., starting a logs loop).
    """
    return OUTPUT_ENABLED
