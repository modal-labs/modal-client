import asyncio
from typing import Any, Callable, Dict, List


def asgi_app_wrapper(asgi_app):
    async def fn(scope, body=None):
        messages_from_app: List[Dict[str, Any]] = []

        # TODO: send disconnect at some point.
        messages_to_app: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        await messages_to_app.put({"type": "http.request", "body": body})

        async def send(message):
            messages_from_app.append(message)

        async def receive():
            return await messages_to_app.get()

        await asgi_app(scope, receive, send)

        return messages_from_app

    return fn


def fastAPI_function_wrapper(fn: Callable, method: str):
    """Return a FastAPI app wrapping a function handler."""
    from fastapi import FastAPI

    app = FastAPI()
    app.add_api_route("/", fn, methods=[method])
    return asgi_app_wrapper(app)
