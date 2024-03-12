# Copyright Modal Labs 2022
from __future__ import annotations

import asyncio
import time

from modal import (
    Image,
    Stub,
    Volume,
    asgi_app,
    build,
    current_function_call_id,
    current_input_id,
    enter,
    exit,
    method,
    web_endpoint,
    wsgi_app,
)
from modal.exception import deprecation_warning

SLEEP_DELAY = 0.1

stub = Stub()


@stub.function()
def square(x):
    return x * x


@stub.function()
def ident(x):
    return x


@stub.function()
def delay(t):
    time.sleep(t)
    return t


@stub.function()
async def delay_async(t):
    await asyncio.sleep(t)
    return t


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
    deprecation_warning((2000, 1, 1), "This function is deprecated")
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


@stub.function()
@wsgi_app()
def basic_wsgi_app():
    def simple_app(environ, start_response):
        status = "200 OK"
        headers = [("Content-type", "text/plain; charset=utf-8")]
        body = environ["wsgi.input"].read()

        start_response(status, headers)
        yield b"got body: " + body

    return simple_app


@stub.cls()
class Cls:
    def __init__(self):
        self._k = 11

    @enter()
    def enter(self):
        self._k += 100

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


@stub.cls()
class LifecycleCls:
    """Ensures that {sync,async} lifecycle hooks work with {sync,async} functions."""

    def __init__(self):
        self.events = []

    def _print_at_exit(self):
        import atexit

        atexit.register(lambda: print("[events:" + ",".join(self.events) + "]"))

    @enter()
    def enter_sync(self):
        self.events.append("enter_sync")

    @enter()
    async def enter_async(self):
        self.events.append("enter_async")

    @exit()
    def exit_sync(self):
        self.events.append("exit_sync")

    @exit()
    async def exit_async(self):
        self.events.append("exit_async")

    @method()
    def f_sync(self, print_at_exit: bool):
        if print_at_exit:
            self._print_at_exit()
        self.events.append("f_sync")
        return self.events

    @method()
    async def f_async(self, print_at_exit: bool):
        if print_at_exit:
            self._print_at_exit()
        self.events.append("f_async")
        return self.events


@stub.function()
def check_sibling_hydration(x):
    assert square.is_hydrated
    assert Cls().f.is_hydrated
    assert Cls().web.is_hydrated
    assert Cls().web.web_url
    assert Cls().generator.is_hydrated
    assert Cls().generator.is_generator
    assert fastapi_app.is_hydrated
    assert fun_returning_gen.is_hydrated
    assert fun_returning_gen.is_generator


@stub.cls()
class ParamCls:
    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y

    @method()
    def f(self, z: int):
        return f"{self.x} {self.y} {z}"

    @method()
    def g(self, z):
        return self.f.local(z)


@stub.function(allow_concurrent_inputs=5)
def sleep_700_sync(x):
    time.sleep(0.7)
    return x * x, current_input_id(), current_function_call_id()


@stub.function(allow_concurrent_inputs=5)
async def sleep_700_async(x):
    await asyncio.sleep(0.7)
    return x * x, current_input_id(), current_function_call_id()


def unassociated_function(x):
    return 100 - x


class BaseCls:
    @enter()
    def enter(self):
        self.x = 2

    @method()
    def run(self, y):
        return self.x * y


@stub.cls()
class DerivedCls(BaseCls):
    pass


@stub.function()
def cube(x):
    # Note: this ends up calling the servicer fixture,
    # which always just returns the sum of the squares of the inputs,
    # regardless of the actual funtion.
    assert square.is_hydrated
    return square.remote(x) * x


@stub.function()
def function_calling_method(x, y, z):
    obj = ParamCls(x, y)
    return obj.f.remote(z)


image = Image.debian_slim().pip_install("xyz")
other_image = Image.debian_slim().pip_install("abc")
volume = Volume.new()
other_volume = Volume.new()


@stub.function(image=image, volumes={"/tmp/xyz": volume})
def check_dep_hydration(x):
    assert image.is_hydrated
    assert other_image.is_hydrated
    assert volume.is_hydrated
    assert other_volume.is_hydrated


@stub.cls()
class BuildCls:
    def __init__(self):
        self._k = 1

    @enter()
    def enter1(self):
        self._k += 10

    @build()
    def build1(self):
        self._k += 100
        return self._k

    @build()
    def build2(self):
        self._k += 1000
        return self._k

    @exit()
    def exit1(self):
        raise Exception("exit called!")

    @method()
    def f(self, x):
        return self._k * x


@stub.cls(enable_memory_snapshot=True)
class CheckpointingCls:
    def __init__(self):
        self._vals = []

    @enter(snap=True)
    def enter1(self):
        self._vals.append("A")

    @enter(snap=True)
    def enter2(self):
        self._vals.append("B")

    @enter()
    def enter3(self):
        self._vals.append("C")

    @method()
    def f(self, x):
        return "".join(self._vals) + x


@stub.cls()
class EventLoopCls:
    @enter()
    async def enter(self):
        self.loop = asyncio.get_running_loop()

    @method()
    async def f(self):
        return self.loop.is_running()
