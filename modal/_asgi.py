from typing import Callable


def asgi_app_wrapper(asgi_app):
    async def fn(scope, body=None):
        messages = []

        async def send(message):
            messages.append(message)

        async def receive():
            return {"type": "http.request", "body": body}

        await asgi_app(scope, receive, send)

        return messages

    return fn


def fastAPI_function_wrapper(fn: Callable, method: str):
    """Return a FastAPI app wrapping a function handler."""
    from fastapi import FastAPI

    app = FastAPI()
    app.add_api_route("/", fn, methods=[method])
    return asgi_app_wrapper(app)
