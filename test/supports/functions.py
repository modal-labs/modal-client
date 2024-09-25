# Copyright Modal Labs 2022
import asyncio
import contextlib
import time
from typing import List, Tuple

from modal import (
    App,
    Sandbox,
    asgi_app,
    batched,
    build,
    current_function_call_id,
    current_input_id,
    enter,
    exit,
    is_local,
    method,
    web_endpoint,
    wsgi_app,
)
from modal.exception import deprecation_warning
from modal.experimental import get_local_input_concurrency, set_local_input_concurrency

SLEEP_DELAY = 0.1

app = App()


@app.function()
def square(x):
    return x * x


@app.function()
def ident(x):
    return x


@app.function()
def delay(t):
    time.sleep(t)
    return t


@app.function()
async def delay_async(t):
    await asyncio.sleep(t)
    return t


@app.function()
async def async_cancel_doesnt_reraise(t):
    try:
        await asyncio.sleep(t)
    except asyncio.CancelledError:
        pass


@app.function()
async def square_async(x):
    await asyncio.sleep(SLEEP_DELAY)
    return x * x


@app.function()
def raises(x):
    raise Exception("Failure!")


@app.function()
def raises_sysexit(x):
    raise SystemExit(1)


@app.function()
def raises_keyboardinterrupt(x):
    raise KeyboardInterrupt()


@app.function()
def gen_n(n):
    for i in range(n):
        yield i**2


@app.function()
def gen_n_fail_on_m(n, m):
    for i in range(n):
        if i == m:
            raise Exception("bad")
        yield i**2


def deprecated_function(x):
    deprecation_warning((2000, 1, 1), "This function is deprecated")
    return x**2


@app.function()
@web_endpoint()
def webhook(arg="world"):
    return {"hello": arg}


def stream():
    for i in range(10):
        time.sleep(SLEEP_DELAY)
        yield f"{i}..."


@app.function()
@web_endpoint()
def webhook_streaming():
    from fastapi.responses import StreamingResponse

    return StreamingResponse(stream())


async def stream_async():
    for i in range(10):
        await asyncio.sleep(SLEEP_DELAY)
        yield f"{i}..."


@app.function()
@web_endpoint()
async def webhook_streaming_async():
    from fastapi.responses import StreamingResponse

    return StreamingResponse(stream_async())


if __name__ == "__main__":
    raise Exception("This line is not supposed to be reachable")


def gen(n):
    for i in range(n):
        yield i**2


@app.function(is_generator=True)
def fun_returning_gen(n):
    return gen(n)


@app.function()
@asgi_app()
def fastapi_app():
    from fastapi import FastAPI

    web_app = FastAPI()

    @web_app.get("/foo")
    async def foo(arg="world"):
        return {"hello": arg}

    return web_app


lifespan_global_asgi_app_func: List[str] = []


@app.function()
@asgi_app()
def fastapi_app_with_lifespan():
    from fastapi import FastAPI, Request

    assert len(lifespan_global_asgi_app_func) == 0

    @contextlib.asynccontextmanager
    async def lifespan(wapp: FastAPI):
        lifespan_global_asgi_app_func.append("enter")
        yield {"foo": "this was set from state"}
        lifespan_global_asgi_app_func.append("exit")

    web_app = FastAPI(lifespan=lifespan)

    @web_app.get("/")
    async def foo(request: Request):
        lifespan_global_asgi_app_func.append("foo")
        return request.state.foo

    return web_app


@app.function()
@asgi_app()
def fastapi_app_with_lifespan_failing_startup():
    from fastapi import FastAPI

    @contextlib.asynccontextmanager
    async def lifespan(wapp: FastAPI):
        print("enter")
        raise Exception("Error while setting up asgi app")
        yield
        print("exit")

    web_app = FastAPI(lifespan=lifespan)

    @web_app.get("/")
    async def foo():
        print("foo")
        return "bar"

    return web_app


@app.function()
@asgi_app()
def fastapi_app_with_lifespan_failing_shutdown():
    from fastapi import FastAPI

    @contextlib.asynccontextmanager
    async def lifespan(wapp: FastAPI):
        print("enter")
        yield
        raise Exception("Error while setting up asgi app")
        print("exit")

    web_app = FastAPI(lifespan=lifespan)

    @web_app.get("/")
    async def foo():
        print("foo")
        return "bar"

    return web_app


lifespan_global_asgi_app_cls: List[str] = []


@app.cls(container_idle_timeout=300, concurrency_limit=1, allow_concurrent_inputs=100)
class fastapi_class_multiple_asgi_apps_lifespans:
    def __init__(self):
        assert len(lifespan_global_asgi_app_cls) == 0

    @asgi_app()
    def my_app1(self):
        from fastapi import FastAPI

        @contextlib.asynccontextmanager
        async def lifespan1(wapp):
            lifespan_global_asgi_app_cls.append("enter1")
            yield
            lifespan_global_asgi_app_cls.append("exit1")

        web_app1 = FastAPI(lifespan=lifespan1)

        @web_app1.get("/")
        async def foo1():
            lifespan_global_asgi_app_cls.append("foo1")
            return "foo1"

        return web_app1

    @asgi_app()
    def my_app2(self):
        from fastapi import FastAPI

        @contextlib.asynccontextmanager
        async def lifespan2(wapp):
            lifespan_global_asgi_app_cls.append("enter2")
            yield
            lifespan_global_asgi_app_cls.append("exit2")

        web_app2 = FastAPI(lifespan=lifespan2)

        @web_app2.get("/")
        async def foo2():
            lifespan_global_asgi_app_cls.append("foo2")
            return "foo2"

        return web_app2

    @exit()
    def exit(self):
        lifespan_global_asgi_app_cls.append("exit")


lifespan_global_asgi_app_cls_fail: List[str] = []


@app.cls(container_idle_timeout=300, concurrency_limit=1, allow_concurrent_inputs=100)
class fastapi_class_lifespan_shutdown_failure:
    def __init__(self):
        assert len(lifespan_global_asgi_app_cls_fail) == 0

    @asgi_app()
    def my_app1(self):
        from fastapi import FastAPI

        @contextlib.asynccontextmanager
        async def lifespan1(wapp):
            lifespan_global_asgi_app_cls_fail.append("enter")
            yield
            raise

        web_app1 = FastAPI(lifespan=lifespan1)

        @web_app1.get("/")
        async def foo():
            lifespan_global_asgi_app_cls_fail.append("foo")
            return "foo"

        return web_app1

    @exit()
    def exit(self):
        lifespan_global_asgi_app_cls_fail.append("lifecycle exit")


@app.function()
@asgi_app()
def non_lifespan_asgi():
    async def app(scope, receive, send):
        if not scope["type"] == "http":
            return

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                ],
            }
        )

        await send(
            {
                "type": "http.response.body",
                "body": b'"foo"',
            }
        )

    return app


@app.function()
@asgi_app()
def error_in_asgi_setup():
    raise Exception("Error while setting up asgi app")


@app.function()
@wsgi_app()
def basic_wsgi_app():
    def simple_app(environ, start_response):
        status = "200 OK"
        headers = [("Content-type", "text/plain; charset=utf-8")]
        body = environ["wsgi.input"].read()

        start_response(status, headers)
        yield b"got body: " + body

    return simple_app


@app.cls()
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

    @asgi_app()
    def asgi_web(self):
        from fastapi import FastAPI

        k_at_construction = self._k  # expected to be 111
        hydrated_at_contruction = square.is_hydrated
        web_app = FastAPI()

        @web_app.get("/")
        def k(arg: str):
            return {
                "at_construction": k_at_construction,
                "at_runtime": self._k,
                "arg": arg,
                "other_hydrated": hydrated_at_contruction,
            }

        return web_app

    def _generator(self, x):
        yield x**3

    @method(is_generator=True)
    def generator(self, x):
        return self._generator(x)


@app.cls()
class LifecycleCls:
    """Ensures that {sync,async} lifecycle hooks work with {sync,async} functions."""

    def __init__(
        self,
        print_at_exit: bool = False,
        sync_enter_duration=0,
        async_enter_duration=0,
        sync_exit_duration=0,
        async_exit_duration=0,
    ):
        self.events: List[str] = []
        self.sync_enter_duration = sync_enter_duration
        self.async_enter_duration = async_enter_duration
        self.sync_exit_duration = sync_exit_duration
        self.async_exit_duration = async_exit_duration
        if print_at_exit:
            self._print_at_exit()

    def _print_at_exit(self):
        import atexit

        atexit.register(lambda: print("[events:" + ",".join(self.events) + "]"))

    @enter()
    def enter_sync(self):
        self.events.append("enter_sync")
        time.sleep(self.sync_enter_duration)

    @enter()
    async def enter_async(self):
        self.events.append("enter_async")
        await asyncio.sleep(self.async_enter_duration)

    @exit()
    def exit_sync(self):
        self.events.append("exit_sync")
        time.sleep(self.sync_exit_duration)

    @exit()
    async def exit_async(self):
        self.events.append("exit_async")
        await asyncio.sleep(self.async_exit_duration)

    @method()
    def local(self):
        self.events.append("local")

    @method()
    def f_sync(self):
        self.events.append("f_sync")
        self.local.local()
        return self.events

    @method()
    async def f_async(self):
        self.events.append("f_async")
        self.local.local()
        return self.events

    @method()
    def delay(self, duration: float):
        self._print_at_exit()
        self.events.append("delay")
        time.sleep(duration)
        return self.events

    @method()
    async def delay_async(self, duration: float):
        self._print_at_exit()
        self.events.append("delay_async")
        await asyncio.sleep(duration)
        return self.events


@app.function()
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


@app.cls()
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


@app.function(allow_concurrent_inputs=5)
def sleep_700_sync(x):
    time.sleep(0.7)
    return x * x, current_input_id(), current_function_call_id()


@app.function(allow_concurrent_inputs=5)
async def sleep_700_async(x):
    await asyncio.sleep(0.7)
    return x * x, current_input_id(), current_function_call_id()


@app.function()
@batched(max_batch_size=4, wait_ms=500)
def batch_function_sync(x: Tuple[int], y: Tuple[int]):
    outputs = []
    for x_i, y_i in zip(x, y):
        outputs.append(x_i / y_i)
    return outputs


@app.function()
@batched(max_batch_size=4, wait_ms=500)
def batch_function_outputs_not_list(x: Tuple[int], y: Tuple[int]):
    return str(x)


@app.function()
@batched(max_batch_size=4, wait_ms=500)
def batch_function_outputs_wrong_len(x: Tuple[int], y: Tuple[int]):
    return list(x) + [0]


@app.function()
@batched(max_batch_size=4, wait_ms=500)
async def batch_function_async(x: Tuple[int], y: Tuple[int]):
    outputs = []
    for x_i, y_i in zip(x, y):
        outputs.append(x_i / y_i)
    await asyncio.sleep(0.1)
    return outputs


def unassociated_function(x):
    return 100 - x


class BaseCls:
    @enter()
    def enter(self):
        self.x = 2

    @method()
    def run(self, y):
        return self.x * y


@app.cls()
class DerivedCls(BaseCls):
    pass


@app.function()
def cube(x):
    # Note: this ends up calling the servicer fixture,
    # which always just returns the sum of the squares of the inputs,
    # regardless of the actual funtion.
    assert square.is_hydrated
    return square.remote(x) * x


@app.function()
def function_calling_method(x, y, z):
    obj = ParamCls(x, y)
    return obj.f.remote(z)


@app.cls()
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


@app.cls(enable_memory_snapshot=True)
class SnapshottingCls:
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


@app.function(enable_memory_snapshot=True)
def snapshotting_square(x):
    return x * x


@app.cls()
class EventLoopCls:
    @enter()
    async def enter(self):
        self.loop = asyncio.get_running_loop()

    @method()
    async def f(self):
        return self.loop.is_running()


@app.function()
def sandbox_f(x):
    # TODO(erikbern): maybe inside containers, `app=app` should be automatic?
    sb = Sandbox.create("echo", str(x), app=app)
    return sb.object_id


@app.function()
def is_local_f(x):
    return is_local()


@app.function()
def raise_large_unicode_exception():
    byte_str = (b"k" * 120_000_000) + b"\x99"
    byte_str.decode("utf-8")


@app.function()
def get_input_concurrency(timeout: int):
    time.sleep(timeout)
    return get_local_input_concurrency()


@app.function()
def set_input_concurrency(start: float):
    set_local_input_concurrency(3)
    time.sleep(1)
    return time.time() - start
