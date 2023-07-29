# Copyright Modal Labs 2022
from __future__ import annotations

import asyncio
import time
from datetime import date

from modal import Stub, asgi_app, method, web_endpoint
from modal.exception import deprecation_warning

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


@stub.function()
@web_endpoint()
def webhook(arg="world"):
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


@stub.cls()
class ParamCls:
    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y

    @method()
    def f(self, z: int):
        return f"{self.x} {self.y} {z}"
