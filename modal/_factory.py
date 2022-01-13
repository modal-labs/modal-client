import functools
import inspect

from ._function_utils import FunctionInfo
from ._session_singleton import (
    get_container_session,
    get_default_session,
    get_running_session,
)


class BaseFactory:
    pass


def make_factory(cls):
    class Factory(cls, BaseFactory):
        """Acts as a wrapper for a transient Object.

        Puts a tag and optionally a session on it. Otherwise just "steals" the object id from the
        underlying object at construction time.
        """

        def __init__(self, fun, session, args_and_kwargs=None):  # TODO: session?
            functools.update_wrapper(self, fun)
            self._fun = fun
            self._args_and_kwargs = args_and_kwargs
            self.function_info = FunctionInfo(fun)

            # This is the only place where tags are being set on objects,
            # besides Function
            tag = self.function_info.get_tag(args_and_kwargs)
            cls._init_static(self, session=session, tag=tag)

        async def _create_impl(self, session):
            if get_container_session() is not None:
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
            object_id = await session.create_object(obj)
            # Note that we can "steal" the object id from the other object
            # and set it on this object. This is a general trick we can do
            # to other objects too.
            return object_id

        def __call__(self, *args, **kwargs):
            """Binds arguments to this object."""
            assert self._args_and_kwargs is None
            return Factory(self._fun, self._session, args_and_kwargs=(args, kwargs))

    Factory.__module__ = cls.__module__
    Factory.__qualname__ = cls.__qualname__ + ".Factory"
    Factory.__doc__ = "\n\n".join(filter(None, [Factory.__doc__, cls.__doc__]))
    return Factory


def make_shared_object_factory_class(cls):
    class SharedObjectFactory(cls, BaseFactory):
        def __init__(self, session, label, namespace):
            self.label = label
            self.namespace = namespace
            tag = f"SHARE({label}, {namespace})"  # TODO: use functioninfo later
            cls._init_static(self, session=session, tag=tag)

        async def _create_impl(self, session):
            object_id = await session.use_object(self.label, self.namespace)
            return object_id

    # TODO: set a bunch of stuff
    return SharedObjectFactory
