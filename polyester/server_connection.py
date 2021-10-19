import asyncio
import json
import urllib.parse
import uuid

import grpc
from grpc.aio._channel import Channel

from .async_utils import retry
from .config import config, logger
from .proto import api_pb2


class BasicAuth(grpc.AuthMetadataPlugin):
    # See https://www.grpc.io/docs/guides/auth/
    def __init__(self, client_type, credentials):
        if credentials and client_type == api_pb2.ClientType.CLIENT:
            token_id, token_secret = credentials
            self._metadata = tuple(
                [
                    ("x-polyester-token-id", token_id),
                    ("x-polyester-token-secret", token_secret),
                ]
            )
        elif credentials and client_type == api_pb2.ClientType.CONTAINER:
            task_id, task_secret = credentials
            self._metadata = tuple(
                [
                    ("x-polyester-task-id", task_id),
                    ("x-polyester-task-secret", task_secret),
                ]
            )
        else:
            self._metadata = tuple()

    def __call__(self, context, callback):
        # TODO: we really shouldn't send the id/secret here, but instead sign the requests
        callback(self._metadata, None)


class GRPCConnectionFactory:
    """Manages gRPC connection with the server. This factory is used by the channel pool."""

    def __init__(self, server_url, client_type=None, credentials=None):
        try:
            o = urllib.parse.urlparse(server_url)
        except Exception:
            logger.info(f"server url: {server_url}")
            raise

        self.target = o.netloc

        basic_auth = BasicAuth(client_type, credentials)
        # TODO: we should make it possible to use tokens with http too, for testing purposes
        if o.scheme.endswith("s"):
            logger.debug("Connecting to %s using secure channel" % o.netloc)
            self.credentials = grpc.composite_channel_credentials(
                grpc.ssl_channel_credentials(),
                grpc.metadata_call_credentials(basic_auth),
            )._credentials
        else:
            logger.debug("Connecting to %s using insecure channel" % o.netloc)
            self.credentials = None

        self.options = [
            ("grpc.max_send_message_length", 1 << 26),
            ("grpc.max_receive_message_length", 1 << 26),
        ]

    async def create(self):
        # Note that the grpc.aio documentation uses secure_channel and insecure_channel, but those are just
        # thin wrappers around the underlying Channel constructor, and insecure_channel currently doesn't
        # let you provide a credentials object
        return Channel(
            target=self.target,
            options=self.options,
            credentials=self.credentials,
            compression=None,
            interceptors=None,
        )
