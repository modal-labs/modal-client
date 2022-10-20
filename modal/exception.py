# Copyright Modal Labs 2022
import sys
import warnings


class Error(Exception):
    """Base error class for all Modal errors"""


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


_INTERNAL_MODULES = ["modal", "synchronicity"]


def _is_internal_frame(frame):
    module = frame.f_globals["__name__"].split(".")[0]
    return module in _INTERNAL_MODULES


def deprecation_warning(msg):
    """Utility for getting the proper stack entry

    See the implementation of the built-in warnings.warn
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

    # This is a lower-level function that warnings.warn uses
    warnings.warn_explicit(msg, DeprecationError, filename, lineno)
