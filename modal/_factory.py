import functools
import inspect

from modal_utils.async_utils import synchronize_apis, synchronizer

from ._app_singleton import get_container_app
from ._function_utils import FunctionInfo


class Factory:
    pass


def make_user_factory(cls):
    print("creating a factory inheriting from", cls)

    class _UserFactory(cls, Factory):  # type: ignore
        """Acts as a wrapper for a transient Object.

        Conceptually a factory "steals" the object id from the
        underlying object at construction time.
        """

        def __init__(self, fun, args_and_kwargs=None):
            functools.update_wrapper(self, fun)
            self._fun = fun
            self._args_and_kwargs = args_and_kwargs
            self.function_info = FunctionInfo(fun)

            # This is the only place where tags are being set on objects,
            # besides Function
            tag = self.function_info.get_tag(args_and_kwargs)
            cls._init_static(self, tag=tag)

        async def load(self, app):
            if get_container_app() is not None:
                assert False

            if self._args_and_kwargs is not None:
                args, kwargs = self._args_and_kwargs
                obj = self._fun(*args, **kwargs)
            else:
                obj = self._fun()
            if inspect.iscoroutine(obj):
                obj = await obj
            if not isinstance(obj, cls):
                raise TypeError(f"expected {obj} to have type {cls}")
            # This is super hacky, but self._fun arguably gets run on the
            # _wrong_ event loop. It's "user code", but it gets executed
            # inside synchronized code. Later, we need some special construct
            # to run user code. For now, we do this dumb translation thing:
            obj = synchronizer._translate_in(obj)
            # Then let's create the object
            object_id = await app.create_object(obj)
            # Note that we can "steal" the object id from the other object
            # and set it on this object. This is a general trick we can do
            # to other objects too.
            return object_id

        def __call__(self, *args, **kwargs):
            """Binds arguments to this object."""
            assert self._args_and_kwargs is None
            return _UserFactory(self._fun, args_and_kwargs=(args, kwargs))

    synchronize_apis(
        _UserFactory, cls.__qualname__ + ".UserFactory", cls.__qualname__ + ".AioUserFactory"
    )  # Needed to create interfaces
    _UserFactory.__module__ = cls.__module__
    _UserFactory.__qualname__ = cls.__qualname__ + "._UserFactory"
    _UserFactory.__doc__ = "\n\n".join(filter(None, [_UserFactory.__doc__, cls.__doc__]))
    return _UserFactory


def make_shared_object_factory_class(cls):
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
    synchronize_apis(
        _SharedObjectFactory, "SharedObjectFactory", "AioSharedObjectFactory"
    )  # Needed to create interfaces
    return _SharedObjectFactory
