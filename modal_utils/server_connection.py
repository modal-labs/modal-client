import urllib.parse
from typing import Iterable, Tuple

import grpc
from grpc.aio._channel import Channel

from modal_proto import api_pb2

from .logger import logger


class BasicAuth(grpc.AuthMetadataPlugin):
    # See https://www.grpc.io/docs/guides/auth/
    _metadata: Iterable[Tuple[str, str]]

    def __init__(self, client_type, credentials):
        if credentials and (client_type == api_pb2.CLIENT_TYPE_CLIENT or client_type == api_pb2.CLIENT_TYPE_WEB_SERVER):
            token_id, token_secret = credentials
            self._metadata = (
                ("x-modal-token-id", token_id),
                ("x-modal-token-secret", token_secret),
            )
        elif credentials and client_type == api_pb2.CLIENT_TYPE_CONTAINER:
            task_id, task_secret = credentials
            self._metadata = (
                ("x-modal-task-id", task_id),
                ("x-modal-task-secret", task_secret),
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
        is_tls = o.scheme.endswith("s")
        host = o.netloc.split(":")[0]
        if credentials or is_tls:
            basic_auth = BasicAuth(client_type, credentials)
            if is_tls:
                channel_credentials = grpc.ssl_channel_credentials()
            else:
                assert host == "localhost", f"TLS should be enabled for gRPC target {self.target}"
                channel_credentials = grpc.local_channel_credentials()
            call_credentials = grpc.metadata_call_credentials(basic_auth)
            self.credentials = grpc.composite_channel_credentials(
                channel_credentials,
                call_credentials,
            )._credentials
        else:
            self.credentials = None

        logger.debug(
            f"Connecting to {self.target} using scheme {o.scheme}, credentials? {self.credentials is not None}"
        )

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
