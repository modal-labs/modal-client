# Copyright Modal Labs 2023

from modal import method, web_endpoint
from modal._utils.function_utils import FunctionInfo, method_has_params


def hasarg(a):
    ...


def noarg():
    ...


def defaultarg(a="hello"):
    ...


def wildcard_args(*wildcard_list, **wildcard_dict):
    ...


def test_is_nullary():
    assert not FunctionInfo(hasarg).is_nullary()
    assert FunctionInfo(noarg).is_nullary()
    assert FunctionInfo(defaultarg).is_nullary()
    assert FunctionInfo(wildcard_args).is_nullary()


class Cls:
    def foo(self):
        pass

    def bar(self, x):
        pass

    def buz(self, *args):
        pass


def test_method_has_params():
    assert not method_has_params(Cls.foo)
    assert not method_has_params(Cls().foo)
    assert method_has_params(Cls.bar)
    assert method_has_params(Cls().bar)
    assert method_has_params(Cls.buz)
    assert method_has_params(Cls().buz)


class Foo:
    def __init__(self):
        pass

    @method()
    def bar(self):
        return "hello"

    @web_endpoint()
    def web(self):
        pass
