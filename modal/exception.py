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
