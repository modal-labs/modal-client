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

    def __init__(self, args=None, session_tag=None):
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

        self.session_tag = session_tag

        # Default values for non-joined objects
        self.joined = False
        self.object_id = None
        self.client = None
        self.session = None
        self.join_lock = None

    async def _join(self):
        raise NotImplementedError

    async def join(self, client, session):
        """Returns a new object that has the properties `client`, `session`, and `object_id` set."""

        if self.object_id is not None:
            return self

        if self.session_tag is not None and self.session_tag in session.objects_by_tag:
            obj = session.objects_by_tag[self.session_tag]
            logger.debug(f"Waiting for lock for object w/ tag {self.session_tag}")
            async with obj.join_lock:
                pass
            logger.debug(f"Acquired lock for object w/ tag {self.session_tag}")
            return obj

        # Note that the lock logic rests on the assumption that the code between here and the next
        # lock acquisition is completely await-free, or else we would introduce a race condition.

        # This is where a bit of magic happens. Since objects are fairly "thin", we can clone them
        # cheaply. What we do here is we create a *new* object that is resolved to an object on
        # the server side. This might be a new object if the object didn't exist, or an existing
        # object: it's up the the subclass to define a _join method that takes care of this.

        logger.debug(f"Joining {self} with tag {self.session_tag}")

        # TODO 1: we should check the session locally to see if it already has resolved this object
        # TODO 2: we should use a mutex to prevent an object from being joined twice simultaneously
        # TODO 3: if the object has a persisted tag then we shouldn't need the session parameter
        # TODO 4: if the object has a persisted tag then we should cache it on the Client

        # Cloning magic:
        cls = type(self)
        obj = cls.__new__(cls)
        obj.args = self.args
        obj.joined = True
        obj.client = client
        obj.session = session
        obj.session_tag = self.session_tag
        obj.join_lock = asyncio.Lock()
        if self.session_tag is not None:
            session.objects_by_tag[self.session_tag] = obj
        async with obj.join_lock:
            obj.object_id = await obj._join()
        return obj

    # def __setattr__(self, k, v):
    #    if k not in ["object_id", "args", "join_lock"]:
    #        raise AttributeError(f"Cannot set attribute {k}")
    #    self.__dict__[k] = v


async def _join_with_defaults(obj):
    # TODO: get rid of these imports - rn it's circular that's why
    from .client import Client
    from .session import Session

    client = await Client.current()
    session = await Session.current()
    return await obj.join(client, session)


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
            # Coroutine fn that returns a generator
            if inspect.iscoroutinefunction(method):
                async for ret in await method(self, *args, **kwargs):
                    yield ret
            else:
                async for ret in method(self, *args, **kwargs):
                    yield ret
        else:
            # Join the object
            new_self = await _join_with_defaults(self)

            # Call the method on the joined object instead
            new_method = getattr(new_self, method.__name__)

            async for ret in new_method(*args, **kwargs):
                yield ret

    return wrapped_method
