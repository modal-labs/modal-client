# Copyright Modal Labs 2023
"""Client for Modal relay servers, allowing users to expose TLS."""

import contextlib
from dataclasses import dataclass
from typing import AsyncIterator

from grpclib import GRPCError, Status

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api

from .client import _Client
from .config import config
from .exception import InvalidError, RemoteError


@dataclass
class Tunnel:
    """A port forwarded from within a running Modal container. Created by `modal.forward()`.

    This is an EXPERIMENTAL API and may change in the future.
    """

    host: str

    @property
    def url(self) -> str:
        """Get the public HTTPS URL of the forwarded port."""
        return f"https://{self.host}"


@contextlib.asynccontextmanager
async def _forward(port: int) -> AsyncIterator[Tunnel]:
    """Expose a port publicly from inside a running Modal container, with TLS.

    This is an EXPERIMENTAL API and may change in the future.

    **Usage:**

    ```python
    from flask import Flask


    app = Flask(__name__)

    @app.route("/")
    def hello_world():
        return "Hello, World!"


    @stub.function()
    def run_app():
        # Start a web server inside the container at port 8000. `modal.forward(8000)` lets us
        # expose that port to the world at a random HTTPS URL.
        with modal.forward(8000) as tunnel:
            print("Server listening at", tunnel.url)
            app.run("0.0.0.0", 8000)
    ```
    """
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise InvalidError(f"Invalid port number {port}")

    if not config.get("task_id"):
        raise InvalidError("Forwarding ports only works inside a Modal container")

    client = await _Client.from_env()

    try:
        response = await client.stub.TunnelStart(api_pb2.TunnelStartRequest(port=port))
    except GRPCError as exc:
        if exc.status == Status.ALREADY_EXISTS:
            raise InvalidError(f"Port {port} is already forwarded")
        elif exc.status == Status.UNAVAILABLE:
            raise RemoteError("Relay server is unavailable") from exc
        else:
            raise

    try:
        yield Tunnel(response.host)
    finally:
        await client.stub.TunnelStop(api_pb2.TunnelStopRequest(port=port))


forward = synchronize_api(_forward)
