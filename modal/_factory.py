import functools
import inspect

from modal_utils.async_utils import synchronize_apis, synchronizer

from ._app_singleton import get_container_app
from ._function_utils import FunctionInfo


class Factory:
    pass


def _local_construction_make(app, cls, fun):
    class _UserFactory(cls, Factory):  # type: ignore
        """Acts as a wrapper for a transient Object.

        Conceptually a factory "steals" the object id from the
        underlying object at construction time.
        """

        def __init__(self):
            # This is the only place where tags are being set on objects,
            # besides Function
            tag = FunctionInfo(fun).get_tag(None)
            cls._init_static(self, tag=tag)
            if get_container_app() is None:
                # Don't do anything inside the container
                app._register_object(self)

        async def load(self, app):
            if get_container_app() is not None:
                assert False
            obj = fun(app)
            if inspect.iscoroutine(obj):
                obj = await obj
            # This is super hacky, but self._fun arguably gets run on the
            # _wrong_ event loop. It's "user code", but it gets executed
            # inside synchronized code. Later, we need some special construct
            # to run user code. For now, we do this dumb translation thing:
            obj = synchronizer._translate_in(obj)
            if not isinstance(obj, cls):
                raise TypeError(f"expected {obj} to have type {cls}")
            # Then let's create the object
            object_id = await app.create_object(obj)
            # Note that we can "steal" the object id from the other object
            # and set it on this object. This is a general trick we can do
            # to other objects too.
            return object_id

    synchronize_apis(_UserFactory)
    return _UserFactory()


def _local_construction(app, cls):
    """Used as a decorator."""
    return functools.partial(_local_construction_make, app, cls)


def make_shared_object_factory_class(cls):
    # TODO: deprecated, replace this with some sort of special reference tag
    class _SharedObjectFactory(cls, Factory):  # type: ignore
        def __init__(self, app_name, object_label, namespace):
            self.app_name = app_name
            self.object_label = object_label
            self.namespace = namespace
            tag = f"#SHARE({app_name}, {object_label}, {namespace})"  # TODO: use functioninfo later
            cls._init_static(self, tag=tag)

        async def load(self, app):
            obj = await app.include(self.app_name, self.object_label, self.namespace)
            return obj.object_id

    # TODO: set a bunch of stuff
    synchronize_apis(_SharedObjectFactory)  # Needed to create interfaces
    return _SharedObjectFactory


def _factory_make(cls, fun):
    # TODO: the FunctionInfo class is a bit overloaded
    # and we should probably factor out the "get_tag" method
    function_info = FunctionInfo(fun)

    class _InternalFactory(cls, Factory):  # type: ignore
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs
            tag = function_info.get_tag((args, kwargs))
            cls._init_static(self, tag=tag)

        async def load(self, app):
            if get_container_app() is not None:
                assert False
            obj = await fun(*self._args, **self._kwargs)
            if not isinstance(obj, cls):
                raise TypeError(f"expected {obj} to have type {cls}")
            object_id = await app.create_object(obj)
            return object_id

    return _InternalFactory


def _factory(cls):
    # Used as a decorator
    return functools.partial(_factory_make, cls)
