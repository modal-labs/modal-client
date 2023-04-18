# Copyright Modal Labs 2022
"""**async interfaces for use with `asyncio` event loops**

The async interfaces are mostly mirrors of the blocking ones, with the `Aio` or `aio_` prefixes.
"""

from .app import AioApp, aio_container_app
from .client import AioClient
from .dict import AioDict
from .functions import (
    AioFunction,
    AioFunctionHandle,
    AioFunctionCall,
    aio_method,
    aio_asgi_app,
    aio_wsgi_app,
    aio_web_endpoint,
)
from .image import AioImage
from .mount import AioMount
from .object import aio_lookup
from .queue import AioQueue
from .secret import AioSecret
from .shared_volume import AioSharedVolume
from .stub import AioStub
from .proxy import AioProxy

__all__ = [
    "AioApp",
    "AioClient",
    "AioDict",
    "AioFunction",
    "AioFunctionCall",
    "AioImage",
    "AioMount",
    "AioQueue",
    "AioSecret",
    "AioSharedVolume",
    "AioStub",
    "AioProxy",
    "AioFunctionHandle",  # noqa
    "aio_container_app",
    "aio_lookup",
    "aio_method",
    "aio_asgi_app",
    "aio_web_endpoint",
    "aio_wsgi_app",
]
