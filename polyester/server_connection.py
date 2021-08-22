import asyncio
import grpc
from grpc.aio._channel import Channel
import json
import uuid

from .async_utils import retry
from .config import config, logger


class BasicAuth(grpc.AuthMetadataPlugin):
    # See https://www.grpc.io/docs/guides/auth/
    def __init__(self, token_id=None, token_secret=None, task_id=None, task_secret=None):
        metadata = tuple([
            ('x-polyester-token-id', token_id),
            ('x-polyester-token-secret', token_secret),
            ('x-polyester-task-id', task_id),
            ('x-polyester-task-secret', task_secret),
        ])
        self._metadata = tuple((k, v) for k, v in metadata if v is not None)

    def __call__(self, context, callback):
        # TODO: we really shouldn't send the id/secret here, but instead sign the requests
        callback(self._metadata, None)


class GRPCConnectionFactory:
    '''Manages gRPC connection with the Server

    TODO: move this elsewhere
    '''
    def __init__(self, server_url, token_id=None, token_secret=None, task_id=None, task_secret=None):
        self._server_url = server_url
        self._token_id = token_id
        self._token_secret = token_secret
        self._task_id = task_id
        self._task_secret = task_secret

    async def create(self):
        protocol, hostname = self._server_url.split('://')
        # TODO: this is a bit janky, we should fix this elsewhere (maybe pass large items by handle instead)
        options = [
            ('grpc.max_send_message_length', 1<<26),
            ('grpc.max_receive_message_length', 1<<26),
        ]
        basic_auth = BasicAuth(self._token_id, self._token_secret, self._task_id, self._task_secret)
        if protocol.endswith('s'):
            logger.debug('Connecting to %s using secure channel' % hostname)
            channel_credentials = grpc.ssl_channel_credentials()
        else:
            logger.debug('Connecting to %s using insecure channel' % hostname)
            channel_credentials = grpc.local_channel_credentials()

        credentials = grpc.composite_channel_credentials(
            channel_credentials,
            grpc.metadata_call_credentials(basic_auth),
        )

        # Note that the grpc.aio documentation uses secure_channel and insecure_channel, but those are just
        # thin wrappers around the underlying Channel constructor, and insecure_channel currently doesn't
        # let you provide a credentials object
        return Channel(
            target=hostname,
            options=options,
            credentials=credentials._credentials,
            compression=None,
            interceptors=None,
        )
