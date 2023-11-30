# Copyright Modal Labs 2022
import random
import signal
import sys
import warnings
from datetime import date


class Error(Exception):
    """
    Base error class for all Modal errors.

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


# TODO(erikbern): we have something similready in _function_utils.py
_INTERNAL_MODULES = ["modal", "modal_utils", "synchronicity"]


def _is_internal_frame(frame):
    module = frame.f_globals["__name__"].split(".")[0]
    return module in _INTERNAL_MODULES


def deprecation_error(deprecated_on: date, msg: str):
    raise DeprecationError(f"Deprecated on {deprecated_on}: {msg}")


def deprecation_warning(deprecated_on: date, msg: str, pending=False):
    """Utility for getting the proper stack entry.

    See the implementation of the built-in [warnings.warn](https://docs.python.org/3/library/warnings.html#available-functions).
    """
    # Find the last non-Modal line that triggered the warning
    try:
        frame = sys._getframe()
        while frame is not None and _is_internal_frame(frame):
            frame = frame.f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
    except ValueError:
        filename = "<unknown>"
        lineno = 0

    warning_cls: type = PendingDeprecationError if pending else DeprecationError

    # This is a lower-level function that warnings.warn uses
    warnings.warn_explicit(f"{deprecated_on}: {msg}", warning_cls, filename, lineno)


def _simulate_preemption_interrupt(signum, frame):
    signal.alarm(30)  # simulate a SIGKILL after 30s
    raise KeyboardInterrupt("Simulated preemption interrupt from modal-client!")


def simulate_preemption(wait_seconds: int, jitter_seconds: int = 0):
    """
    Utility for simulating a preemption interrupt after `wait_seconds` seconds.
    The first interrupt is the SIGINT/SIGTERM signal. After 30 seconds a second
    interrupt will trigger. This second interrupt simulates SIGKILL, and should not be caught.
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
