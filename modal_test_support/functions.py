# Copyright Modal Labs 2022
from __future__ import annotations
import asyncio
from datetime import date
import time

import pytest

from modal import Stub, method, web_endpoint, asgi_app
from modal.exception import DeprecationError, deprecation_warning

SLEEP_DELAY = 0.1

stub = Stub()


@stub.function()
def square(x):
    return x * x


@stub.function()
def delay(t):
    time.sleep(t)


@stub.function()
async def square_async(x):
    await asyncio.sleep(SLEEP_DELAY)
    return x * x


@stub.function()
def raises(x):
    raise Exception("Failure!")


@stub.function()
def raises_sysexit(x):
    raise SystemExit(1)


@stub.function()
def raises_keyboardinterrupt(x):
    raise KeyboardInterrupt()


@stub.function()
def gen_n(n):
    for i in range(n):
        yield i**2


@stub.function()
def gen_n_fail_on_m(n, m):
    for i in range(n):
        if i == m:
            raise Exception("bad")
        yield i**2


def deprecated_function(x):
    deprecation_warning(date(2000, 1, 1), "This function is deprecated")
    return x**2


# TODO(erikbern): once we remove support for the "old" lifecycle method syntax,
# let's consolidate some tests - Cube, CubeAsync, and Cls pretty much test
# the same things (but currently only Cls uses the "new" syntax).


with pytest.warns(DeprecationError):

    class Cube:
        _events: list[str] = []

        def __init__(self):
            self._events.append("init")

        def __enter__(self):
            self._events.append("enter")

        def __exit__(self, typ, exc, tb):
            self._events.append("exit")

        @stub.function()  # Will trigger deprecation warning
        def f(self, x):
            self._events.append("call")
            return x**3


with pytest.warns(DeprecationError):

    class CubeAsync:
        _events: list[str] = []

        def __init__(self):
            self._events.append("init")

        async def __aenter__(self):
            self._events.append("enter")

        async def __aexit__(self, typ, exc, tb):
            self._events.append("exit")

        @stub.function()  # Will trigger deprecation warning
        async def f(self, x):
            self._events.append("call")
            return x**3


@stub.function()
@web_endpoint()
def webhook(arg="world"):
    return {"hello": arg}


with pytest.warns(DeprecationError):

    @stub.webhook
    def webhook_old(arg="world"):
        return {"hello": arg}


with pytest.warns(DeprecationError):

    class WebhookLifecycleClass:
        _events: list[str] = []

        def __init__(self):
            self._events.append("init")

        async def __aenter__(self):
            self._events.append("enter")

        async def __aexit__(self, typ, exc, tb):
            self._events.append("exit")

        @stub.function()  # Will trigger deprecation warning
        @web_endpoint()
        def webhook(self, arg="world"):
            self._events.append("call")
            return {"hello": arg}


def stream():
    for i in range(10):
        time.sleep(SLEEP_DELAY)
        yield f"{i}..."


@stub.function()
@web_endpoint()
def webhook_streaming():
    from fastapi.responses import StreamingResponse

    return StreamingResponse(stream())


async def stream_async():
    for i in range(10):
        await asyncio.sleep(SLEEP_DELAY)
        yield f"{i}..."


@stub.function()
@web_endpoint()
async def webhook_streaming_async():
    from fastapi.responses import StreamingResponse

    return StreamingResponse(stream_async())


if __name__ == "__main__":
    raise Exception("This line is not supposed to be reachable")


def gen(n):
    for i in range(n):
        yield i**2


@stub.function(is_generator=True)
def fun_returning_gen(n):
    return gen(n)


@stub.function()
@asgi_app()
def fastapi_app():
    from fastapi import FastAPI

    web_app = FastAPI()

    @web_app.get("/foo")
    async def foo(arg="world"):
        return {"hello": arg}

    return web_app


@stub.cls()
class Cls:
    def __enter__(self):
        self._k = 111

    @method()
    def f(self, x):
        return self._k * x

    @web_endpoint()
    def web(self, arg):
        return {"ret": arg * self._k}

    def _generator(self, x):
        yield x**3

    @method(is_generator=True)
    def generator(self, x):
        return self._generator(x)


@stub.function()
def check_sibling_hydration(x):
    assert square.is_hydrated()
    assert Cls().f.is_hydrated()
    assert Cls().web.is_hydrated()
    assert Cls().web.web_url
    assert Cls().generator.is_hydrated()
    assert Cls().generator.is_generator
    assert fastapi_app.is_hydrated()
    assert fun_returning_gen.is_hydrated()
    assert fun_returning_gen.is_generator
