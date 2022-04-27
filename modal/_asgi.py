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
    """Take in a function that's not attached to an ASGI app,
    and return a wrapped FastAPI app with this function as the root handler."""
    from fastapi import FastAPI

    app = FastAPI()

    # Using app.route() directly seems to not set up the FastAPI models correctly.
    if method == "POST":
        app.post("/")(fn)
    elif method == "GET":
        app.get("/")(fn)
    elif method == "DELETE":
        app.delete("/")(fn)
    elif method == "HEAD":
        app.head("/")(fn)
    elif method == "OPTIONS":
        app.options("/")(fn)
    elif method == "PATCH":
        app.patch("/")(fn)
    elif method == "PUT":
        app.put("/")(fn)

    return asgi_app_wrapper(app)
