# Copyright Modal Labs 2022
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
        with stub.run():
            f.call()
    except modal.Error:
        # Catch any exception raised by Modal's systems.
        print("Responding to error...")
    ```
    """


class RemoteError(Error):
    """Raised when an error occurs on the Modal server."""


class TimeoutError(Error):
    """Raised when a Function exceeds its execution duration limit and times out."""


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
