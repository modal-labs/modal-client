import functools
import inspect
import json

import synchronicity

from modal_utils.async_utils import synchronize_apis, synchronizer

from ._app_singleton import get_container_app
from ._function_utils import FunctionInfo


class Factory:
    pass


def _create_callback(fun):
    # This is a bit of an ugly hack, but we need to know what interface the
    # user function will use to return objects (eg it might return a sync
    # version of some object, and we want to convert it to an internal type).
    # We infer it from the function signature.
    if inspect.iscoroutinefunction(fun):
        interface = synchronicity.Interface.ASYNC
    elif inspect.isfunction(fun):
        interface = synchronicity.Interface.BLOCKING
    else:
        raise Exception(f"{fun}: expected function but got {type(fun)}")

    # Create a coroutine we can use internally
    return synchronizer.create_callback(fun, interface)


def _local_construction_make(app, cls, fun):
    callback = _create_callback(fun)

    class _UserFactory(cls, Factory):  # type: ignore
        """Acts as a wrapper for a transient Object.

        Conceptually a factory "steals" the object id from the
        underlying object at construction time.
        """

        def __init__(self):
            # This is the only place where tags are being set on objects,
            # besides Function
            tag = FunctionInfo(fun).get_tag()
            cls._init_static(self, tag=tag)
            if get_container_app() is None:
                # Don't do anything inside the container
                app._register_object(self)

        async def load(self, app):
            if get_container_app() is not None:
                assert False
            obj = await callback()
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
        def __init__(self, app, app_name, object_label, namespace):
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
    # TODO: we should add support for user code:
    # callback = _create_callback(fun)

    class _InternalFactory(cls, Factory):  # type: ignore
        def __init__(self, app, **kwargs):
            self._kwargs = kwargs
            tag = FunctionInfo(fun).get_tag()

            # Append the arguments (but not the app) to the tag
            fun_app_bound = functools.partial(fun, None)
            signature = inspect.signature(fun_app_bound)
            args = signature.bind(**kwargs)
            args.apply_defaults()
            args_list = list(args.arguments.values())[1:]  # remove app
            args_str = json.dumps(args_list)[1:-1]  # remove the enclosing []
            tag = f"{tag}({args_str})"

            cls._init_static(self, tag=tag)

        async def load(self, app):
            if get_container_app() is not None:
                assert False
            obj = await fun(app, **self._kwargs)
            if not isinstance(obj, cls):
                raise TypeError(f"expected {obj} to have type {cls}")
            object_id = await app.create_object(obj)
            return object_id

    return _InternalFactory


def _factory(cls):
    # Used as a decorator
    return functools.partial(_factory_make, cls)
