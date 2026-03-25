# Copyright Modal Labs 2024
"""Interface to Modal's OutputManager functionality.

These functions live here so that Modal library code can import them without
transitively importing Rich, as we do in _output/rich.py. This allows
us to avoid importing Rich for client code that runs in the container environment.
"""

import contextlib
from collections.abc import Generator

from ._output.manager import OutputManager

# Re-export OutputManager for external use
__all__ = ["enable_output", "OutputManager"]


@contextlib.contextmanager
def enable_output() -> Generator[None, None, None]:
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
    previous_output_manager = OutputManager.get()

    from ._output.rich import RichOutputManager

    new_manager = RichOutputManager()
    OutputManager._set(new_manager)
    try:
        yield
    finally:
        OutputManager._set(previous_output_manager)
