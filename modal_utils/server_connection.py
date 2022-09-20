import urllib.parse
from typing import Dict

import grpclib.events
from grpclib.client import Channel

from modal._tracing import inject_tracing_context
from modal_proto import api_pb2

from .logger import logger


def auth_metadata(client_type: api_pb2.ClientType, credentials) -> Dict[str, str]:
    if credentials and (client_type == api_pb2.CLIENT_TYPE_CLIENT or client_type == api_pb2.CLIENT_TYPE_WEB_SERVER):
        token_id, token_secret = credentials
        return {
            "x-modal-token-id": token_id,
            "x-modal-token-secret": token_secret,
        }
    elif credentials and client_type == api_pb2.CLIENT_TYPE_CONTAINER:
        task_id, task_secret = credentials
        return {
            "x-modal-task-id": task_id,
            "x-modal-task-secret": task_secret,
        }
    else:
        return {}


class GRPCConnectionFactory:
    """Manages gRPC connection with the server. This factory is used by the channel pool."""

    def __init__(self, server_url: str, client_type: api_pb2.ClientType = None, credentials=None) -> None:
        try:
            o = urllib.parse.urlparse(server_url)
        except Exception:
            logger.exception(f"failed to parse server url: {server_url}")
            raise

        self.target = o.netloc
        self.is_tls = o.scheme.endswith("s")

        self.metadata = auth_metadata(client_type, credentials)
        logger.debug(f"Connecting to {self.target} using scheme {o.scheme}")

    def create(self) -> Channel:
        parts = self.target.split(":")
        assert len(parts) <= 2, "Invalid target location: " + self.target
        channel = Channel(
            host=parts[0],
            port=parts[1] if len(parts) == 2 else 443 if self.is_tls else 80,
            ssl=self.is_tls,
        )

        # Inject metadata for the client.
        async def send_request(event: grpclib.events.SendRequest) -> None:
            for k, v in self.metadata.items():
                event.metadata[k] = v

            inject_tracing_context(event.metadata)

        grpclib.events.listen(channel, grpclib.events.SendRequest, send_request)
        return channel
