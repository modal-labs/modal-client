# Potentially temporary file as we try to isolate Rich from code that runs in a container
import contextlib
from typing import Generator


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
