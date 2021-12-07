# TODO(erikbern): this is a bit if a dumb hack to avoid a circular dependency
# the Object class needs to get the session singleton, but session.py imports object.py
# Let's figure out a cleaner way to do this later, this is just a stopgap thing.

_singleton = None


def get_session_singleton():
    return _singleton


def set_session_singleton(s):
    global _singleton
    _singleton = s
