# TODO(erikbern): this is a bit if a dumb hack to avoid a circular dependency
# the Object class needs to get the app singleton, but app.py imports object.py
# Let's figure out a cleaner way to do this later, this is just a stopgap thing.

_container_app = None


def get_container_app():
    return _container_app


def set_container_app(a):
    global _container_app
    _container_app = a


# These are only useful on the client, not in the container. When in the
# container, the default app and the running app will both equal to the
# container singleton.

_running_app = None


def get_running_app():
    return _running_app


def set_running_app(a):
    global _running_app
    if _running_app is not None and a is not None:
        raise RuntimeError("Cannot run 2 modal apps at once")
    _running_app = a


_default_app = None


def get_default_app():
    global _running_app
    if _running_app is not None:
        return _running_app

    global _default_app
    if _default_app is None:
        # Local import to break import loop
        from .app import App

        _default_app = App()
    return _default_app


def set_default_app(a):
    global _default_app
    _default_app = a
