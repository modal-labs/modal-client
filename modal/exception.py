import sys
import warnings


class Error(Exception):
    """Base error class for all Modal errors"""


class RemoteError(Error):
    """An error was raised on the Modal server."""


class AuthError(Error):
    """The client has missing or invalid authentication."""


class ConnectionError(Error):
    """An issue was raised while connecting to the Modal servers."""


class InvalidError(Error):
    """Used when user does something invalid."""


class VersionError(Error):
    """The current client version of Modal is unsupported."""


class NotFoundError(Error):
    """A requested resource was not found."""


class ExecutionError(Error):
    """Something unexpected happen during runtime."""


class DeprecationError(UserWarning):
    """Overloading it to evade the default filter, which excludes __main__."""


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
