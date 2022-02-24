class RemoteError(Exception):
    pass


class AuthError(Exception):
    pass


class ConnectionError(Exception):
    pass


class InvalidError(Exception):
    """Used when user does something invalid."""


class VersionError(Exception):
    pass


class NotFoundError(Exception):
    def __init__(self, msg, obj_repr):
        super(NotFoundError, self).__init__(msg)
        self.obj_repr = obj_repr


class ExecutionError(Exception):
    pass
