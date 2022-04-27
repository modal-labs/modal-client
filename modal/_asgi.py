from typing import Callable, List


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


def fastAPI_function_wrapper(fn: Callable, methods: List[str]):
    """Take in a function that's not attached to an ASGI app,
    and return a wrapped FastAPI app with this function as the root handler."""
    from fastapi import FastAPI

    app = FastAPI()

    if len(methods) == 1 and methods[0] == "POST":
        # Using app.route() directly seems to not use the Pydantic models correctly.
        app.post("/")(fn)
    else:
        app.route("/", methods)(fn)
    return asgi_app_wrapper(app)
