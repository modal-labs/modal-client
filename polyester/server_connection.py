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


MAX_MESSAGE_LENGTH = 1000000000  # 100 MB


class GRPCConnectionFactory:
    """Manages gRPC connection with the server. This factory is used by the channel pool."""

    def __init__(self, server_url, client_type=None, credentials=None):
        try:
            o = urllib.parse.urlparse(server_url)
        except Exception:
            logger.info(f"server url: {server_url}")
            raise

        self.target = o.netloc

        host = o.netloc.split(":")[0]
        if credentials and not o.scheme.endswith("s") and host != "localhost":
            # There are only two options for vanilla http traffic in GRPC:
            # - grpc.experimental.insecure_channel(): can't be used with call credentials
            # - grpc.local_channel_credentials(): can only be used with localhost
            # The problem is inside containers, we connect to host.docker.internal, so
            # we need to use the insecure channel, which means we can't use call credentials.
            credentials = None

        if credentials:
            basic_auth = BasicAuth(client_type, credentials)
            if o.scheme.endswith("s"):
                channel_credentials = grpc.ssl_channel_credentials()
            else:
                channel_credentials = grpc.local_channel_credentials()
            call_credentials = grpc.metadata_call_credentials(basic_auth)
            self.credentials = grpc.composite_channel_credentials(
                channel_credentials,
                call_credentials,
            )._credentials
        else:
            self.credentials = None

        self.options = [
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
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
