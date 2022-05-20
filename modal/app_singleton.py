import synchronicity

from modal_utils.async_utils import synchronizer

# TODO(erikbern): this is a bit if a dumb hack to avoid a circular dependency
# the Object class needs to get the app singleton, but app.py imports object.py
# Let's figure out a cleaner way to do this later, this is just a stopgap thing.

_container_app = None  # _RunningApp
container_app = None  # RunningApp
aio_container_app = None  # AioRunningApp


def set_container_app(running_app):
    global _container_app, container_app, aio_container_app
    _container_app = running_app
    container_app = synchronizer._translate_out(running_app, synchronicity.Interface.BLOCKING)
    aio_container_app = synchronizer._translate_out(running_app, synchronicity.Interface.ASYNC)
