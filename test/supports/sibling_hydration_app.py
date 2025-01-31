import modal
from modal import asgi_app, enter, method, web_endpoint

app = modal.App()


@app.function()
def square(x):
    return x * x


@app.function()
@asgi_app()
def fastapi_app():
    return None


def gen():
    yield


@app.function(is_generator=True)
def fun_returning_gen():
    return gen()


@app.function()
def function_calling_method(x, y, z):
    obj = ParamCls(x, y)
    return obj.f.remote(z)


@app.function()
def check_sibling_hydration(x):
    assert square.is_hydrated
    assert fastapi_app.is_hydrated
    assert fastapi_app.web_url
    assert fun_returning_gen.is_hydrated
    assert fun_returning_gen.is_generator

    # make sure the underlying service function for the class is hydrated:
    assert NonParamCls._get_class_service_function().is_hydrated  # type: ignore
    assert ParamCls._get_class_service_function().is_hydrated  # type: ignore

    # notably not hydrated at this point:
    # NonParamCls()  (instance of parameter-less class - note that hydration shouldn't require any roundtrips for this)
    # NonParamCls().f  (method of parameter-less class - note that hydration shouldn't require any roundtrips for this)
    # ParamCls(x=1, y=3)  (parameter-bound class instance)


@app.cls()
class NonParamCls:
    _k = 11  # not a parameter, just a static initial value

    @enter()
    def enter(self):
        self._k = 111

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
class ParamCls:
    x: int = modal.parameter()
    y: str = modal.parameter()

    @method()
    def f(self, z: int):
        return f"{self.x} {self.y} {z}"

    @method()
    def g(self, z):
        return self.f.local(z)
