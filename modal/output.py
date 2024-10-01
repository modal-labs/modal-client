# Copyright Modal Labs 2024
"""Interface to Modal's OutputManager functionality.

These functions live here so that Modal library code can import them without
transitively importing Rich, as we do in global scope in _output.py. This allows
us to avoid importing Rich for client code that runs in the container environment.

"""
import contextlib
from typing import TYPE_CHECKING, Generator, Optional

if TYPE_CHECKING:
    from ._output import OutputManager


@contextlib.contextmanager
def enable_output(show_progress: bool = True) -> Generator[None, None, None]:
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
    from ._output import OutputManager

    with OutputManager.enable_output(show_progress):
        yield


def _get_output_manager() -> Optional["OutputManager"]:
    """Interface to the OutputManager with a deferred import."""
    from ._output import OutputManager

    # This will return None when output has not been enabled,
    # as should generally be the case when using Modal in a container
    return OutputManager.get()
