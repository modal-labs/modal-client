# Copyright Modal Labs 2023
"""Client for Modal relay servers, allowing users to expose TLS.

Based on the Rust implementation in our internal modal-relay crate. That one is
end-to-end tested and is the source of truth for our client protocol.
"""

import asyncio
import contextlib
import json
import ssl
from dataclasses import dataclass
from typing import Any, AsyncIterator, Tuple

import certifi
from google.protobuf.empty_pb2 import Empty

from modal_proto import api_pb2
from modal_utils.async_utils import TaskContext, synchronize_api
from modal_utils.grpc_utils import retry_transient_errors

from .client import _Client
from .config import config, logger
from .exception import InvalidError

ssl_context = ssl.create_default_context(cafile=certifi.where())


async def control_send(writer: asyncio.StreamWriter, obj: Any) -> None:
    writer.write(json.dumps(obj).encode() + b"\0")
    await writer.drain()


async def control_recv(reader: asyncio.StreamReader) -> Any:
    try:
        data = await reader.readuntil(b"\0")
    except asyncio.IncompleteReadError:
        return None
    return json.loads(data[:-1].decode("utf-8"))


@dataclass
class RelayClient:
    host: str
    port: int
    task_id: str
    task_secret: str

    async def start_relay(self, forwarded_host: str, forwarded_port: int) -> Tuple["RelayDriver", str]:
        reader, writer = await asyncio.open_connection(self.host, self.port, ssl=ssl_context)
        await control_send(writer, {"HelloNew": {"task_id": self.task_id, "task_secret": self.task_secret}})

        resp = await control_recv(reader)
        host = resp["Hello"]["host"]
        token = resp["Hello"]["token"]

        driver = RelayDriver(self, reader, writer, token, forwarded_host, forwarded_port)
        return driver, host


class RelayDriver:
    def __init__(
        self,
        client: RelayClient,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        token: str,
        forwarded_host: str,
        forwarded_port: int,
    ) -> None:
        self.client = client
        self.reader = reader
        self.writer = writer
        self.token = token
        self.forwarded_host = forwarded_host
        self.forwarded_port = forwarded_port
        self.task_context = TaskContext(grace=1)

    async def start(self) -> None:
        await self.task_context.start()
        self.task_context.create_task(self._run())

    async def stop(self) -> None:
        await self.task_context.stop()
        self.writer.close()

    async def _run(self) -> None:
        """Drive the TCP proxy, reconnecting on any errors."""
        while True:
            try:
                await self._control()
            except asyncio.CancelledError:
                raise  # Do not catch CancelledError
            except Exception as exc:
                logger.debug("reconnecting due to error in control loop", exc_info=exc)

            self.writer.close()
            while True:
                try:
                    await self._reconnect()
                    break
                except asyncio.CancelledError:
                    raise  # Do not catch CancelledError
                except Exception as exc:
                    logger.debug("error reconnecting to server", exc_info=exc)
                    await asyncio.sleep(1)

    async def _reconnect(self) -> None:
        reader, writer = await asyncio.open_connection(self.client.host, self.client.port, ssl=ssl_context)
        await control_send(writer, {"HelloReconnect": {"token": self.token}})
        resp = await control_recv(reader)
        self.token = resp["hello"]["token"]
        self.reader = reader
        self.writer = writer

    async def _control(self) -> None:
        while True:
            resp = await control_recv(self.reader)
            if resp is None:
                break
            if "Forward" in resp:
                conn: str = resp["Forward"]["conn"]
                logger.debug(f"new relay connection, forwarding to {self.forwarded_host}:{self.forwarded_port}")
                self.task_context.create_task(self._new_stream(conn))

    async def _new_stream(self, conn: str) -> None:
        reader, writer = await asyncio.open_connection(self.client.host, self.client.port, ssl=ssl_context)
        await control_send(writer, {"Accept": {"conn": conn}})
        local_reader, local_writer = await asyncio.open_connection(self.forwarded_host, self.forwarded_port)
        task1 = self.task_context.create_task(_asyncio_copy(reader, local_writer))
        task2 = self.task_context.create_task(_asyncio_copy(local_reader, writer))
        await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)
        task1.cancel()
        task2.cancel()
        writer.close()
        local_writer.close()


async def _asyncio_copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Pump bytes from a StreamReader to a StreamWriter."""
    while True:
        try:
            data = await reader.read(8192)
        except ConnectionError:
            break
        if not data:  # EOF
            break
        writer.write(data)
        await writer.drain()


@dataclass
class Tunnel:
    host: str

    @property
    def url(self) -> str:
        return f"https://{self.host}"


@contextlib.asynccontextmanager
async def _forward(port: int) -> AsyncIterator[Tunnel]:
    """Forward a port from within a running Modal container."""
    client = await _Client.from_env()

    response: api_pb2.RelayListResponse = await retry_transient_errors(client.stub.RelayList, Empty())

    # TODO: connect to multiple hostnames concurrently for fault tolerance,
    # and race them to find the one with lowest latency
    hostname = response.hostnames[0]

    task_id, task_secret = config.get("task_id"), config.get("task_secret")
    if not task_id or not task_secret:
        raise InvalidError("task_id and task_secret must be set in config")
    relay_client = RelayClient(hostname, 443, task_id, task_secret)
    driver, relay_host = await relay_client.start_relay("localhost", port)
    await driver.start()
    try:
        yield Tunnel(relay_host)
    finally:
        await driver.stop()


forward = synchronize_api(_forward)
