# TODO(erikbern): this is a bit if a dumb hack to avoid a circular dependency
# the Object class needs to get the app singleton, but app.py imports object.py
# Let's figure out a cleaner way to do this later, this is just a stopgap thing.

_container_app = None  # RunningApp


def get_container_app():
    return _container_app


def set_container_app(a):
    global _container_app
    _container_app = a
