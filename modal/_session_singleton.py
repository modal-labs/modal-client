# TODO(erikbern): this is a bit if a dumb hack to avoid a circular dependency
# the Object class needs to get the session singleton, but session.py imports object.py
# Let's figure out a cleaner way to do this later, this is just a stopgap thing.

_container_session = None


def get_container_session():
    return _container_session


def set_container_session(s):
    global _container_session
    _container_session = s


# These are only useful on the client, not in the container. When in the
# container, the default session and the running session will both equal to the
# container singleton.

_running_session = None


def set_running_session(s):
    global _running_session
    if _running_session is not None and s is not None:
        raise RuntimeError("Cannot run 2 modal sessions at once")
    _running_session = s


_default_session = None


def get_default_session():
    global _running_session
    if _running_session is not None:
        return _running_session

    global _default_session
    if _default_session is None:
        # Local import to break import loop
        from .session import Session

        _default_session = Session()
    return _default_session


def set_default_session(s):
    global _default_session
    _default_session = s
