class RemoteError(Exception):
    """An error was raised on the Modal server."""


class AuthError(Exception):
    """The client has missing or invalid authentication."""


class ConnectionError(Exception):
    """An issue was raised while connecting to the Modal servers."""


class InvalidError(Exception):
    """Used when user does something invalid."""


class VersionError(Exception):
    """The current client version of Modal is unsupported."""


class NotFoundError(Exception):
    """A requested resource was not found."""


class ExecutionError(Exception):
    """Something unexpected happen during runtime."""
