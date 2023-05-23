# Copyright Modal Labs 2022
"""**async interfaces for use with `asyncio` event loops**

The async interfaces are mostly mirrors of the blocking ones, with the `Aio` or `aio_` prefixes.
"""

from modal.exception import deprecation_warning
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
from .queue import AioQueue
from .secret import AioSecret
from .shared_volume import AioSharedVolume
from .stub import AioStub
from .proxy import AioProxy
import datetime

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
    "aio_method",
    "aio_asgi_app",
    "aio_web_endpoint",
    "aio_wsgi_app",
]

deprecation_warning(
    datetime.date(2023, 5, 12),
    "The `modal.aio` module and `Aio*` classes will soon be deprecated.\n"
    "For calling functions asynchronously, use `await some_function.aio(...)`\n"
    "Instead of separate classes for async usage, the interface now only changes how to call the methods."
    "Where you would have previously used `await AioDict.lookup(...)` you now use "
    "`await Dict.lookup.aio(...)` instead.\nObjects that are themselves generators or context managers "
    "now conform to both the blocking and async interfaces, and returned objects of all functions/methods",
    pending=True,
)
