# Copyright Modal Labs 2022
from __future__ import annotations
import asyncio
import time

from modal import Stub
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


@stub.generator()
def gen_n(n):
    for i in range(n):
        yield i**2


@stub.generator()
def gen_n_fail_on_m(n, m):
    for i in range(n):
        if i == m:
            raise Exception("bad")
        yield i**2


def deprecated_function(x):
    deprecation_warning("This function is deprecated")
    return x**2


class Cube:
    _events: list[str] = []

    def __init__(self):
        self._events.append("init")

    def __enter__(self):
        self._events.append("enter")

    def __exit__(self, typ, exc, tb):
        self._events.append("exit")

    @stub.function()
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

    @stub.function()
    async def f(self, x):
        self._events.append("call")
        return x**3


@stub.webhook
def webhook(arg="world"):
    return f"Hello, {arg}"


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
        return f"Hello, {arg}"


if __name__ == "__main__":
    raise Exception("This line is not supposed to be reachable")
