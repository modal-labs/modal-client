# Copyright Modal Labs 2022
import random
import signal
import sys
import warnings
from datetime import date
from typing import Tuple


class Error(Exception):
    """
    Base class for all Modal errors. See [`modal.exception`](/docs/reference/modal.exception) for the specialized
    error classes.

    **Usage**

    ```python notest
    import modal

    try:
        ...
    except modal.Error:
        # Catch any exception raised by Modal's systems.
        print("Responding to error...")
    ```
    """


class RemoteError(Error):
    """Raised when an error occurs on the Modal server."""


class TimeoutError(Error):
    """Base class for Modal timeouts."""


class SandboxTimeoutError(TimeoutError):
    """Raised when a Sandbox exceeds its execution duration limit and times out."""


class SandboxTerminatedError(Error):
    """Raised when a Sandbox is terminated for an internal reason."""


class FunctionTimeoutError(TimeoutError):
    """Raised when a Function exceeds its execution duration limit and times out."""


class MountUploadTimeoutError(TimeoutError):
    """Raised when a Mount upload times out."""


class VolumeUploadTimeoutError(TimeoutError):
    """Raised when a Volume upload times out."""


class InteractiveTimeoutError(TimeoutError):
    """Raised when interactive frontends time out while trying to connect to a container."""


class AuthError(Error):
    """Raised when a client has missing or invalid authentication."""


class ConnectionError(Error):
    """Raised when an issue occurs while connecting to the Modal servers."""


class InvalidError(Error):
    """Raised when user does something invalid."""


class VersionError(Error):
    """Raised when the current client version of Modal is unsupported."""


class NotFoundError(Error):
    """Raised when a requested resource was not found."""


class ExecutionError(Error):
    """Raised when something unexpected happened during runtime."""


class DeprecationError(UserWarning):
    """UserWarning category emitted when a deprecated Modal feature or API is used."""

    # Overloading it to evade the default filter, which excludes __main__.


class PendingDeprecationError(UserWarning):
    """Soon to be deprecated feature. Only used intermittently because of multi-repo concerns."""


class _CliUserExecutionError(Exception):
    """mdmd:hidden
    Private wrapper for exceptions during when importing or running stubs from the CLI.

    This intentionally does not inherit from `modal.exception.Error` because it
    is a private type that should never bubble up to users. Exceptions raised in
    the CLI at this stage will have tracebacks printed.
    """

    def __init__(self, user_source: str):
        # `user_source` should be the filepath for the user code that is the source of the exception.
        # This is used by our exception handler to show the traceback starting from that point.
        self.user_source = user_source


# TODO(erikbern): we have something similready in function_utils.py
_INTERNAL_MODULES = ["modal", "synchronicity"]


def _is_internal_frame(frame):
    module = frame.f_globals["__name__"].split(".")[0]
    return module in _INTERNAL_MODULES


def deprecation_error(deprecated_on: Tuple[int, int, int], msg: str):
    raise DeprecationError(f"Deprecated on {date(*deprecated_on)}: {msg}")


def deprecation_warning(
    deprecated_on: Tuple[int, int, int], msg: str, pending: bool = False, show_source: bool = True
) -> None:
    """Utility for getting the proper stack entry.

    See the implementation of the built-in [warnings.warn](https://docs.python.org/3/library/warnings.html#available-functions).
    """
    filename, lineno = "<unknown>", 0
    if show_source:
        # Find the last non-Modal line that triggered the warning
        try:
            frame = sys._getframe()
            while frame is not None and _is_internal_frame(frame):
                frame = frame.f_back
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
        except ValueError:
            # Use the defaults from above
            pass

    warning_cls: type = PendingDeprecationError if pending else DeprecationError

    # This is a lower-level function that warnings.warn uses
    warnings.warn_explicit(f"{date(*deprecated_on)}: {msg}", warning_cls, filename, lineno)


def _simulate_preemption_interrupt(signum, frame):
    signal.alarm(30)  # simulate a SIGKILL after 30s
    raise KeyboardInterrupt("Simulated preemption interrupt from modal-client!")


def simulate_preemption(wait_seconds: int, jitter_seconds: int = 0):
    """
    Utility for simulating a preemption interrupt after `wait_seconds` seconds.
    The first interrupt is the SIGINT signal. After 30 seconds, a second
    interrupt will trigger.

    This second interrupt simulates SIGKILL, and should not be caught.
    Optionally add between zero and `jitter_seconds` seconds of additional waiting before first interrupt.

    **Usage:**

    ```python notest
    import time
    from modal.exception import simulate_preemption

    simulate_preemption(3)

    try:
        time.sleep(4)
    except KeyboardInterrupt:
        print("got preempted") # Handle interrupt
        raise
    ```

    See https://modal.com/docs/guide/preemption for more details on preemption
    handling.
    """
    signal.signal(signal.SIGALRM, _simulate_preemption_interrupt)
    jitter = random.randrange(0, jitter_seconds) if jitter_seconds else 0
    signal.alarm(wait_seconds + jitter)


class InputCancellation(BaseException):
    """Raised when the current input is cancelled by the task

    Intentionally a BaseException instead of an Exception, so it won't get
    caught by unspecified user exception clauses that might be used for retries and
    other control flow.
    """


class ModuleNotMountable(Exception):
    pass
