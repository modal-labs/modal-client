# Copyright Modal Labs 2025
import struct
import time
from typing import Any, Collection, Generic, Mapping, NewType, Optional, Type, TypeVar, Union, cast

from grpclib.client import Channel as GRPCLibChannel, Stream as GRPCLibStream
from grpclib.const import Cardinality
from grpclib.encoding.base import CodecBase
from grpclib.exceptions import ProtocolError
from grpclib.metadata import Deadline
from grpclib.protocol import Stream as ProtocolStream
from multidict import MultiDict

_Value = Union[str, bytes]
_MetadataLike = Union[Mapping[str, _Value], Collection[tuple[str, _Value]]]
_SendType = TypeVar("_SendType")
_RecvType = TypeVar("_RecvType")
_Metadata = NewType("_Metadata", "MultiDict[_Value]")


async def send_message(
    stream: "ProtocolStream",
    codec: "CodecBase",
    message: Any,
    message_type: Type,
    *,
    end: bool = False,
) -> None:
    reply_bin = codec.encode(message, message_type)
    reply_data = struct.pack("?", False) + struct.pack(">I", len(reply_bin)) + reply_bin
    await stream.send_data(reply_data, end_stream=end)


class Stream(GRPCLibStream, Generic[_SendType, _RecvType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def send_message(
        self,
        message: _SendType,
        *,
        end: bool = False,
    ) -> None:
        if not self._send_request_done:
            await self.send_request()

        end_stream = end
        if not self._cardinality.client_streaming:
            if self._send_message_done:
                raise ProtocolError("Message was already sent")
            else:
                end_stream = True

        if self._end_done:
            raise ProtocolError("Stream is ended")

        with self._wrapper:
            (message,) = await self._dispatch.send_message(message)
            await send_message(self._stream, self._codec, message, self._send_type, end=end_stream)
            self._send_message_done = True
            self._messages_sent += 1
            self._stream.connection.messages_sent += 1
            self._stream.connection.last_message_sent = time.monotonic()
            if end:
                self._end_done = True


class Channel(GRPCLibChannel):
    def request(
        self,
        name: str,
        cardinality: Cardinality,
        request_type: Type[_SendType],
        reply_type: Type[_RecvType],
        *,
        timeout: Optional[float] = None,
        deadline: Optional[Deadline] = None,
        metadata: Optional[_MetadataLike] = None,
    ) -> Stream[_SendType, _RecvType]:
        if timeout is not None and deadline is None:
            deadline = Deadline.from_timeout(timeout)
        elif timeout is not None and deadline is not None:
            deadline = min(Deadline.from_timeout(timeout), deadline)

        metadata = cast(_Metadata, MultiDict(metadata or ()))

        return Stream(
            self,
            name,
            metadata,
            cardinality,
            request_type,
            reply_type,
            codec=self._codec,
            status_details_codec=self._status_details_codec,
            dispatch=self.__dispatch__,
            deadline=deadline,
        )
