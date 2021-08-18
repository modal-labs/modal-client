import asyncio
import grpc
import json
import uuid

from .async_utils import retry
from .grpc_utils import BasicInterceptor, make_interceptors
from .config import config, logger


class BasicAuthInterceptor(BasicInterceptor):
    def __init__(self, token_id=None, token_secret=None, task_id=None, task_secret=None):
        self._metadata = {
            'x-polyester-token-id': token_id,
            'x-polyester-token-secret': token_secret,
            'x-polyester-task-id': task_id,
            'x-polyester-task-secret': task_secret
        }

    async def intercept(self, client_call_details):
        # TODO: we really shouldn't send the id/secret here, but instead sign the requests
        for k, v in self._metadata.items():
            if v is not None:
                client_call_details.metadata[k] = v


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
        basic_auth_interceptor = BasicAuthInterceptor(self._token_id, self._token_secret, self._task_id, self._task_secret)
        interceptors = make_interceptors(basic_auth_interceptor)
        if protocol.endswith('s'):
            logger.debug('Connecting to %s using secure channel' % hostname)
            return grpc.aio.secure_channel(hostname, grpc.ssl_channel_credentials(), options=options, interceptors=interceptors)
        else:
            logger.debug('Connecting to %s using insecure channel' % hostname)
            return grpc.aio.insecure_channel(hostname, options=options, interceptors=interceptors)
