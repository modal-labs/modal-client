class RemoteError(Exception):
    pass


class AuthError(Exception):
    pass


class ConnectionError(Exception):
    pass


class InvalidError(Exception):
    """Used when user does something invalid."""

    pass


class VersionError(Exception):
    pass


class NotFoundError(Exception):
    pass


class ExecutionError(Exception):
    pass
