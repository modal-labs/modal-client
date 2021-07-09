import asyncio
import grpc
import json
import uuid

from .async_utils import retry
from .config import config, logger


class GRPCConnectionFactory:
    '''Manages gRPC connection with the Server

    TODO: move this elsewhere
    '''
    def __init__(self, server_url):
        self.server_url = server_url

    async def create(self):
        protocol, hostname = self.server_url.split('://')
        # TODO: this is a bit janky, we should fix this elsewhere (maybe pass large items by handle instead)
        options = [
            ('grpc.max_send_message_length', 1<<26),
            ('grpc.max_receive_message_length', 1<<26),
        ]
        if protocol.endswith('s'):
            logger.debug('Connecting to %s using secure channel' % hostname)
            return grpc.aio.secure_channel(hostname, grpc.ssl_channel_credentials(), options=options)
        else:
            logger.debug('Connecting to %s using insecure channel' % hostname)
            return grpc.aio.insecure_channel(hostname, options=options)
