import asyncio
import functools
import inspect

from .async_utils import synchronizer
from .config import logger


class ObjectMeta(type):
    type_to_name = {}
    name_to_type = {}

    def __new__(metacls, name, bases, dct):
        # Synchronize class
        new_cls = synchronizer.create_class(metacls, name, bases, dct)

        # Register class as serializable
        ObjectMeta.type_to_name[new_cls] = name
        ObjectMeta.name_to_type[name] = new_cls

        logger.debug(f"Created Object class {name}")
        return new_cls


class Args:
    def __init__(self, data):
        self.__dict__["data"] = data if data is not None else {}

    def __getattr__(self, k):
        return self.__dict__["data"][k]

    def __setattr__(self, k, v):
        raise AttributeError("Args object is immutable")


class Object(metaclass=ObjectMeta):
    # A bit ugly to leverage implemenation inheritance here, but I guess you could
    # roughly think of this class as a mixin

    def __init__(self, args=None):
        logger.debug(f"Creating object {self}")

        # TODO: should we make these attributes hidden for subclasses?
        # (i.e. "private" not even "protected" to use the C++ terminology)
        # Feels like there could be some benefits of doing so
        if isinstance(args, dict):
            self.args = Args(args)
        elif isinstance(args, Args):
            self.args = args
        elif args is None:
            self.args = None
        else:
            raise Exception(f"{args} of type {type(args)} must be instance of (dict, Args, NoneType)")

        # Default values for non-created objects
        self.created = False
        self.session = None
        self.tag = None

    async def _create_or_get(self):
        raise NotImplementedError


async def _join_with_defaults(obj):
    # TODO: get rid of these imports - rn it's circular that's why
    from .client import Client
    from .session import DeprecatedSession

    client = await Client.current()
    DEPRECATED_session = await DeprecatedSession.current()
    return await obj.join(client, DEPRECATED_session)


def requires_join(method):
    @functools.wraps(method)
    async def wrapped_method(self, *args, **kwargs):
        if self.joined:
            # Object already has an object id, just keep going
            return await method(self, *args, **kwargs)
        else:
            # Join the object
            new_self = await _join_with_defaults(self)

            # Call the method on the joined object instead
            new_method = getattr(new_self, method.__name__)
            return await new_method(*args, **kwargs)

    return wrapped_method


def requires_join_generator(method):
    @functools.wraps(method)
    async def wrapped_method(self, *args, **kwargs):
        if self.joined:
            result = method(self, *args, **kwargs)
            # Coroutine fn that returns a generator
            if inspect.iscoroutine(result):
                result = await result
            async for ret in result:
                yield ret
        else:
            # Join the object
            new_self = await _join_with_defaults(self)

            # Call the method on the joined object instead
            new_method = getattr(new_self, method.__name__)

            async for ret in new_method(*args, **kwargs):
                yield ret

    return wrapped_method


def requires_create(method):
    @functools.wraps(method)
    def wrapped_method(self, *args, **kwargs):
        if not self.created:
            raise Exception(f"Error running method {method} on object {self}: object is not created yet")
        return method(self, *args, **kwargs)

    return wrapped_method
