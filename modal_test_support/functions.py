# Copyright Modal Labs 2022
from __future__ import annotations
import asyncio
from datetime import date
import time

from modal import Stub
from modal.exception import deprecation_warning

SLEEP_DELAY = 0.1

stub = Stub()


@stub.function
def square(x):
    return x * x


@stub.function
def delay(t):
    time.sleep(t)


@stub.function
async def square_async(x):
    await asyncio.sleep(SLEEP_DELAY)
    return x * x


@stub.function
def raises(x):
    raise Exception("Failure!")


@stub.function
def raises_sysexit(x):
    raise SystemExit(1)


@stub.function
def raises_keyboardinterrupt(x):
    raise KeyboardInterrupt()


@stub.function
def gen_n(n):
    for i in range(n):
        yield i**2


@stub.function
def gen_n_fail_on_m(n, m):
    for i in range(n):
        if i == m:
            raise Exception("bad")
        yield i**2


def deprecated_function(x):
    deprecation_warning(date(2000, 1, 1), "This function is deprecated")
    return x**2


class Cube:
    _events: list[str] = []

    def __init__(self):
        self._events.append("init")

    def __enter__(self):
        self._events.append("enter")

    def __exit__(self, typ, exc, tb):
        self._events.append("exit")

    @stub.function
    def f(self, x):
        self._events.append("call")
        return x**3


class CubeAsync:
    _events: list[str] = []

    def __init__(self):
        self._events.append("init")

    async def __aenter__(self):
        self._events.append("enter")

    async def __aexit__(self, typ, exc, tb):
        self._events.append("exit")

    @stub.function
    async def f(self, x):
        self._events.append("call")
        return x**3


@stub.webhook
def webhook(arg="world"):
    return {"hello": arg}


class WebhookLifecycleClass:
    _events: list[str] = []

    def __init__(self):
        self._events.append("init")

    async def __aenter__(self):
        self._events.append("enter")

    async def __aexit__(self, typ, exc, tb):
        self._events.append("exit")

    @stub.webhook
    def webhook(self, arg="world"):
        self._events.append("call")
        return {"hello": arg}


if __name__ == "__main__":
    raise Exception("This line is not supposed to be reachable")


def gen(n):
    for i in range(n):
        yield i**2


@stub.function(is_generator=True)
def fun_returning_gen(n):
    return gen(n)


@stub.asgi
def fastapi_app():
    from fastapi import FastAPI

    web_app = FastAPI()

    @web_app.get("/foo")
    async def foo(arg="world"):
        return {"hello": arg}

    return web_app
