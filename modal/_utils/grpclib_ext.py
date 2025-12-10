# Copyright Modal Labs 2025
import gzip
import struct
import time
from typing import Any, Collection, Generic, Mapping, Optional, Type, TypeVar, Union

from grpclib.client import Channel as GRPCLibChannel, Stream as GRPCLibStream
from grpclib.const import Cardinality
from grpclib.encoding.base import GRPC_CONTENT_TYPE, CodecBase
from grpclib.exceptions import ProtocolError
from grpclib.metadata import USER_AGENT, Deadline, encode_metadata, encode_timeout
from grpclib.protocol import Stream as ProtocolStream
from multidict import MultiDict

_Value = Union[str, bytes]
_MetadataLike = Union[Mapping[str, _Value], Collection[tuple[str, _Value]]]
_SendType = TypeVar("_SendType")
_RecvType = TypeVar("_RecvType")


async def send_message(
    stream: "ProtocolStream",
    codec: "CodecBase",
    message: Any,
    message_type: Type,
    *,
    end: bool = False,
) -> None:
    reply_bin = codec.encode(message, message_type)
    payload = gzip.compress(reply_bin)
    reply_data = struct.pack("?", True) + struct.pack(">I", len(payload)) + payload
    await stream.send_data(reply_data, end_stream=end)


class Stream(GRPCLibStream, Generic[_SendType, _RecvType]):
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

    async def send_request(self, *, end: bool = False) -> None:
        if self._send_request_done:
            raise ProtocolError("Request is already sent")

        if end and not self._cardinality.client_streaming:
            raise ProtocolError("Unary request requires a message to be sent before ending outgoing stream")

        with self._wrapper:
            protocol = await self._channel.__connect__()
            stream = protocol.processor.connection.create_stream(wrapper=self._wrapper)

            headers = [
                (":method", "POST"),
                (":scheme", self._channel._scheme),
                (":path", self._method_name),
                (":authority", self._channel._authority),
            ]
            if self._deadline is not None:
                timeout = self._deadline.time_remaining()
                headers.append(("grpc-timeout", encode_timeout(timeout)))
            # FIXME: remove this check after this issue gets resolved:
            #   https://github.com/googleapis/googleapis.github.io/issues/27
            if self._codec.__content_subtype__ == "proto":
                content_type = GRPC_CONTENT_TYPE
            else:
                content_type = GRPC_CONTENT_TYPE + "+" + self._codec.__content_subtype__
            headers.extend(
                (
                    ("te", "trailers"),
                    ("content-type", content_type),
                    ("user-agent", USER_AGENT),
                    ("grpc-encoding", "gzip"),
                )
            )
            (metadata,) = await self._dispatch.send_request(
                self._metadata,
                method_name=self._method_name,
                deadline=self._deadline,
                content_type=content_type,
            )
            headers.extend(encode_metadata(metadata))
            release_stream = await stream.send_request(
                headers,
                end_stream=end,
                _processor=protocol.processor,
            )
            self._stream = stream
            self._release_stream = release_stream
            self.peer = self._stream.connection.get_peer()
            self._send_request_done = True
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

        metadata = MultiDict(metadata or ())

        return Stream(
            self,
            name,
            metadata,  # type: ignore
            cardinality,
            request_type,
            reply_type,
            codec=self._codec,
            status_details_codec=self._status_details_codec,
            dispatch=self.__dispatch__,
            deadline=deadline,
        )
