import asyncio
import grpc
import json
import uuid

from .async_utils import retry
from .grpc_utils import BasicInterceptor, make_interceptors
from .config import config, logger


class BasicAuthInterceptor(BasicInterceptor):
    def __init__(self, token_id, token_secret):
        self._token_id = token_id
        self._token_secret = token_secret

    async def intercept(self, client_call_details):
        # TODO: we really shouldn't send the secret here, but instead sign the requests
        if self._token_id is not None:
            client_call_details.metadata['x-polyester-token-id'] = self._token_id
        if self._token_secret is not None:
            client_call_details.metadata['x-polyester-token-secret'] = self._token_secret


class GRPCConnectionFactory:
    '''Manages gRPC connection with the Server

    TODO: move this elsewhere
    '''
    def __init__(self, server_url, token_id=None, token_secret=None):
        self._server_url = server_url
        self._token_id = token_id
        self._token_secret = token_secret

    async def create(self):
        protocol, hostname = self._server_url.split('://')
        # TODO: this is a bit janky, we should fix this elsewhere (maybe pass large items by handle instead)
        options = [
            ('grpc.max_send_message_length', 1<<26),
            ('grpc.max_receive_message_length', 1<<26),
        ]
        interceptors = make_interceptors(BasicAuthInterceptor(self._token_id, self._token_secret))
        if protocol.endswith('s'):
            logger.debug('Connecting to %s using secure channel' % hostname)
            return grpc.aio.secure_channel(hostname, grpc.ssl_channel_credentials(), options=options, interceptors=interceptors)
        else:
            logger.debug('Connecting to %s using insecure channel' % hostname)
            return grpc.aio.insecure_channel(hostname, options=options, interceptors=interceptors)
