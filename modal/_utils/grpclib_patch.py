# This module is a minimum version of https://github.com/vmagamedov/grpclib/blob/master/grpclib/events.py
# that creates patched versions of the events in `grpclib.events` and defines a `patch_grpclib_client_channel`
# `patch_grpclib_server` to patch the dispatcher in place.
import inspect
import sys
from functools import partial
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


async def custom_dispatch(self, event: EventPatchMixin) -> Any:
    # Use the __grpclib_type__ type for finding the listener
    for callback in self._listeners[event.__grpclib_type__]:
        await callback(event)
        if event.__interrupted__:
            break
    return tuple(getattr(event, name) for name in event.__payload__)


# Events common to client and server
async def send_message(message: Any, self) -> Tuple[Any]:
    return await custom_dispatch(
        self,
        PatchedSendMessage(  # type: ignore
            message=message,
        ),
    )


async def recv_message(message: Any, self) -> Tuple[Any]:
    return await custom_dispatch(
        self,
        PatchedSendMessage(  # type: ignore
            message=message,
        ),
    )


# Client only events
async def send_request(
    metadata: Any,
    *,
    method_name: str,
    deadline: Any,
    content_type: str,
    self,
) -> Tuple[Any]:
    return await custom_dispatch(
        self,
        PatchedSendRequest(metadata=metadata, method_name=method_name, deadline=deadline, content_type=content_type),
    )


async def recv_initial_metadata(
    metadata: Any,
    self,
) -> Tuple[Any]:
    return await custom_dispatch(
        self,
        PatchedRecvInitialMetadata(metadata=metadata),
    )


async def recv_trailing_metadata(
    metadata: Any,
    *,
    status: grpclib.const.Status,
    status_message: Optional[str],
    status_details: Any,
    self,
) -> Tuple[Any]:
    return await custom_dispatch(
        self,
        PatchedRecvTrailingMetadata(
            metadata=metadata, status=status, status_message=status_message, status_details=status_details
        ),
    )


# Server events
async def recv_request(
    metadata: Any,
    method_func: Any,
    *,
    method_name: str,
    deadline: Any,
    content_type: str,
    user_agent: Optional[str],
    peer: Any,
    self,
) -> Tuple[Any, Any]:
    return await custom_dispatch(
        self,
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
    metadata: Any,
    self,
) -> Tuple[Any]:
    return await custom_dispatch(
        self,
        PatchedSendInitialMetadata(metadata=metadata),
    )


async def send_trailing_metadata(
    metadata: Any,
    *,
    status: grpclib.const.Status,
    status_message: Optional[str],
    status_details: Any,
    self,
) -> Tuple[Any]:
    return await custom_dispatch(
        self,
        PatchedSendTrailingMetadata(
            metadata=metadata, status=status, status_message=status_message, status_details=status_details
        ),
    )


def patch_grpclib_common(dispatch):
    dispatch.send_message = partial(send_message, self=dispatch)
    dispatch.recv_message = partial(recv_message, self=dispatch)


def patch_grpclib_client_channel(channel: grpclib.client.Channel):
    if PY314:
        dispatch = channel.__dispatch__
        patch_grpclib_common(dispatch)
        dispatch.send_request = partial(send_request, self=dispatch)
        dispatch.recv_initial_metadata = partial(recv_initial_metadata, self=dispatch)
        dispatch.recv_trailing_metadata = partial(recv_trailing_metadata, self=dispatch)


def patch_grpclib_server(server: grpclib.server.Server):
    if PY314:
        dispatch = server.__dispatch__
        patch_grpclib_common(dispatch)

        dispatch.recv_request = partial(recv_request, self=dispatch)
        dispatch.send_initial_metadata = partial(send_initial_metadata, self=dispatch)
        dispatch.send_trailing_metadata = partial(send_trailing_metadata, self=dispatch)
