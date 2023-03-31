# Copyright Modal Labs 2022
import asyncio
import functools
from typing import Any, Callable, Dict

from asgiref.wsgi import WsgiToAsgi

from modal_utils.async_utils import TaskContext


def asgi_app_wrapper(asgi_app):
    async def fn(scope, body=None):
        messages_from_app: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        # TODO: send disconnect at some point.
        messages_to_app: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        await messages_to_app.put({"type": "http.request", "body": body})

        async def send(message):
            await messages_from_app.put(message)

        async def receive():
            return await messages_to_app.get()

        # Run the ASGI app, while draining the send message queue at the same time,
        # and yielding results.
        async with TaskContext(grace=1.0) as tc:
            app_task = tc.create_task(asgi_app(scope, receive, send))

            while True:
                pop_task = tc.create_task(messages_from_app.get())

                try:
                    done, pending = await asyncio.wait([pop_task, app_task], return_when=asyncio.FIRST_COMPLETED)
                except asyncio.CancelledError:
                    break

                if pop_task in done:
                    yield pop_task.result()

                if app_task in done:
                    while not messages_from_app.empty():
                        yield messages_from_app.get_nowait()
                    break

    return fn


def wsgi_app_wrapper(wsgi_app):
    asgi_app = WsgiToAsgi(wsgi_app)
    return asgi_app_wrapper(asgi_app)


def _stream_generator(fn: Callable) -> Callable:
    # Wrap a generator function in a handler that returns a StreamingResponse
    from fastapi.responses import StreamingResponse

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        generator = fn(*args, **kwargs)
        return StreamingResponse(generator)

    return wrapper


def webhook_asgi_app(fn: Callable, method: str, is_generator: bool):
    """Return a FastAPI app wrapping a function handler."""
    # Pulls in `fastapi` module, which is slow to import.
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    if is_generator:
        fn = _stream_generator(fn)

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_api_route("/", fn, methods=[method])
    return app
