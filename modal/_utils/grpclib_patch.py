# Copyright Modal Labs 2025

# This module is a minimum version of https://github.com/vmagamedov/grpclib/blob/master/grpclib/events.py
# that creates patched versions of the events in `grpclib.events` and a `patch_grpclib_client_channel`
# `patch_grpclib_server` to patch the dispatcher.

import inspect
import sys
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, FrozenSet, List, Tuple, Type

import grpclib
import grpclib.client
import grpclib.events
import grpclib.server

PY314 = sys.version_info >= (3, 14)


class EventPatchMixin:
    """Construct object with __slots__ based on the annotations of cls.__grpclib_type__"""

    __slots__: Tuple
    __payload__: Any
    __readonly__: FrozenSet[str]
    __interrupted__: bool
    __grpclib_type__: type

    def __init_subclass__(cls, **kwargs):
        grpclib_type = cls.__grpclib_type__
        if hasattr(inspect, "get_annotations"):
            annotations = inspect.get_annotations(grpclib_type)
        else:
            annotations = grpclib_type.__annotations__

        payload = grpclib_type.__payload__  # type: ignore
        cls.__slots__ = tuple(name for name in annotations)
        cls.__readonly__ = frozenset(name for name in annotations if name not in payload)
        super().__init_subclass__(**kwargs)


class PatchedSendMessage(EventPatchMixin, grpclib.events.SendMessage):
    __grpclib_type__ = grpclib.events.SendMessage


class PatchedRecvMessage(EventPatchMixin, grpclib.events.RecvMessage):
    __grpclib_type__ = grpclib.events.RecvMessage


class PatchedRecvRequest(EventPatchMixin, grpclib.events.RecvRequest):
    __grpclib_type__ = grpclib.events.RecvRequest


class PatchedSendRequest(EventPatchMixin, grpclib.events.SendRequest):
    __grpclib_type__ = grpclib.events.SendRequest


class PatchedRecvInitialMetadata(EventPatchMixin, grpclib.events.RecvInitialMetadata):
    __grpclib_type__ = grpclib.events.RecvInitialMetadata


class PatchedRecvTrailingMetadata(EventPatchMixin, grpclib.events.RecvTrailingMetadata):
    __grpclib_type__ = grpclib.events.RecvTrailingMetadata


class PatchedSendInitialMetadata(EventPatchMixin, grpclib.events.SendInitialMetadata):
    __grpclib_type__ = grpclib.events.SendInitialMetadata


class PatchedSendTrailingMetadata(EventPatchMixin, grpclib.events.SendTrailingMetadata):
    __grpclib_type__ = grpclib.events.SendTrailingMetadata


_Callback = Callable[[Any], Coroutine[Any, Any, None]]


class Dispatcher:
    def __init__(self) -> None:
        self._listeners: Dict[Type, List[_Callback]] = defaultdict(list)

    def add_listener(self, event_type: Type, callback: _Callback) -> None:
        self._listeners[event_type].append(callback)

    async def __dispatch__(self, event: EventPatchMixin) -> Any:
        for callback in self._listeners[event.__grpclib_type__]:
            await callback(event)
            if event.__interrupted__:
                break
        return tuple(getattr(event, name) for name in event.__payload__)


class DispatchCommonEvents(Dispatcher):
    async def send_message(self, message, **kwargs) -> Tuple[Any]:
        return await self.__dispatch__(PatchedSendMessage(message=message, **kwargs))

    async def recv_message(self, message, **kwargs) -> Tuple[Any]:
        return await self.__dispatch__(PatchedRecvMessage(message=message, **kwargs))


class DispatchServerEvents(DispatchCommonEvents):
    async def recv_request(self, metadata, method_func, **kwargs) -> Tuple[Any, Any]:
        return await self.__dispatch__(PatchedRecvRequest(metadata=metadata, method_func=method_func, **kwargs))

    async def send_initial_metadata(self, metadata, **kwargs) -> Tuple[Any]:
        return await self.__dispatch__(PatchedSendInitialMetadata(metadata=metadata, **kwargs))

    async def send_trailing_metadata(self, metadata, **kwargs) -> Tuple[Any]:
        return await self.__dispatch__(PatchedSendTrailingMetadata(metadata=metadata, **kwargs))


class DispatchChannelEvents(DispatchCommonEvents):
    async def send_request(self, metadata, **kwargs) -> Tuple[Any]:
        return await self.__dispatch__(PatchedSendRequest(metadata=metadata, **kwargs))

    async def recv_initial_metadata(self, metadata, **kwargs) -> Tuple[Any]:
        return await self.__dispatch__(PatchedRecvInitialMetadata(metadata=metadata, **kwargs))

    async def recv_trailing_metadata(self, metadata, **kwargs) -> Tuple[Any]:
        return await self.__dispatch__(PatchedRecvTrailingMetadata(metadata=metadata, **kwargs))


def patch_grpclib_client_channel(channel: grpclib.client.Channel):
    """Patches the channels dispatcher with a version that works with Python 3.14."""
    if PY314:
        channel.__dispatch__ = DispatchChannelEvents()  # type: ignore


def patch_grpclib_server(server: grpclib.server.Server):
    """Patches the server dispatcher with a version that works with Python 3.14."""
    if PY314:
        server.__dispatch__ = DispatchServerEvents()  # type: ignore
