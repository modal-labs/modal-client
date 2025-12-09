# This module is a minimum version of https://github.com/vmagamedov/grpclib/blob/master/grpclib/events.py
# that creates patched versions of the events in `grpclib.events` and defines a `patch_grpclib_client_channel`
# `patch_grpclib_server` to patch the dispatcher in place.
import inspect
import sys
from types import MethodType
from typing import Any, Collection, Optional, Tuple

import grpclib
import grpclib.client
import grpclib.events
import grpclib.server

PY314 = sys.version_info >= (3, 14)


class EventPatchMixin:
    """Construct object with __slots__ based on the annotations of cls.__grpclib_type__"""

    __grpclib_type__: type
    __interrupted__: bool
    __payload__: Collection[str]

    def __init_subclass__(cls, **kwargs):
        grpclib_type = cls.__grpclib_type__
        # Patch is only applied in Python 3.14 which has `inspect.get_annotations`
        annotations = inspect.get_annotations(grpclib_type)
        payload = grpclib_type.__payload__
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


async def patched_dispatch(self, event: EventPatchMixin) -> Any:
    # Use the __grpclib_type__ type for finding the listener. This assumes the dispatcher
    # will continue to use `self._listeners` to store all the listeners.
    for callback in self._listeners[event.__grpclib_type__]:
        await callback(event)
        if event.__interrupted__:
            break
    return tuple(getattr(event, name) for name in event.__payload__)


# Events common to client and server
async def send_message(self, message: Any) -> Tuple[Any]:
    return await self.__dispatch__(
        PatchedSendMessage(  # type: ignore
            message=message,
        ),
    )


async def recv_message(self, message: Any) -> Tuple[Any]:
    return await self.__dispatch__(
        PatchedSendMessage(  # type: ignore
            message=message,
        ),
    )


# Client events
async def send_request(
    self,
    metadata: Any,
    *,
    method_name: str,
    deadline: Any,
    content_type: str,
) -> Tuple[Any]:
    return await self.__dispatch__(
        PatchedSendRequest(metadata=metadata, method_name=method_name, deadline=deadline, content_type=content_type),
    )


async def recv_initial_metadata(
    self,
    metadata: Any,
) -> Tuple[Any]:
    return await self.__dispatch__(
        PatchedRecvInitialMetadata(metadata=metadata),
    )


async def recv_trailing_metadata(
    self,
    metadata: Any,
    *,
    status: grpclib.const.Status,
    status_message: Optional[str],
    status_details: Any,
) -> Tuple[Any]:
    return await self.__dispatch__(
        PatchedRecvTrailingMetadata(
            metadata=metadata, status=status, status_message=status_message, status_details=status_details
        ),
    )


# Server events
async def recv_request(
    self,
    metadata: Any,
    method_func: Any,
    *,
    method_name: str,
    deadline: Any,
    content_type: str,
    user_agent: Optional[str],
    peer: Any,
) -> Tuple[Any, Any]:
    return await self.__dispatch__(
        PatchedRecvRequest(
            metadata=metadata,
            method_func=method_func,
            method_name=method_name,
            deadline=deadline,
            content_type=content_type,
            user_agent=user_agent,
            peer=peer,
        ),
    )


async def send_initial_metadata(
    self,
    metadata: Any,
) -> Tuple[Any]:
    return await self.__dispatch__(
        PatchedSendInitialMetadata(metadata=metadata),
    )


async def send_trailing_metadata(
    self,
    metadata: Any,
    *,
    status: grpclib.const.Status,
    status_message: Optional[str],
    status_details: Any,
) -> Tuple[Any]:
    return await self.__dispatch__(
        PatchedSendTrailingMetadata(
            metadata=metadata, status=status, status_message=status_message, status_details=status_details
        ),
    )


def patch_grpclib_common(dispatch):
    dispatch.__dispatch__ = MethodType(patched_dispatch, dispatch)
    dispatch.send_message = MethodType(send_message, dispatch)
    dispatch.recv_message = MethodType(recv_message, dispatch)


def patch_grpclib_client_channel(channel: grpclib.client.Channel):
    if PY314:
        dispatch = channel.__dispatch__
        patch_grpclib_common(dispatch)
        dispatch.send_request = MethodType(send_request, dispatch)
        dispatch.recv_initial_metadata = MethodType(recv_initial_metadata, dispatch)
        dispatch.recv_trailing_metadata = MethodType(recv_trailing_metadata, dispatch)


def patch_grpclib_server(server: grpclib.server.Server):
    if PY314:
        dispatch = server.__dispatch__
        patch_grpclib_common(dispatch)

        dispatch.recv_request = MethodType(recv_request, dispatch)
        dispatch.send_initial_metadata = MethodType(send_initial_metadata, dispatch)
        dispatch.send_trailing_metadata = MethodType(send_trailing_metadata, dispatch)
