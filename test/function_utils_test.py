# Copyright Modal Labs 2023
import pytest
from typing import List

from modal import Queue, method, web_endpoint
from modal._serialization import deserialize
from modal._utils.function_utils import FunctionInfo, get_referred_objects, method_has_params
from modal.exception import InvalidError
from modal.object import Object
from modal.partial_function import _PartialFunction
from modal_proto import api_pb2

q1 = Queue.from_name("q1", create_if_missing=True)
q2 = Queue.from_name("q2", create_if_missing=True)


def f1():
    q1.get()


def f2():
    f1()
    q2.get()


def test_referred_objects():
    objs: List[Object] = get_referred_objects(f1)
    assert objs == [q1]


def test_referred_objects_recursive():
    objs: List[Object] = get_referred_objects(f2)
    assert set(objs) == set([q1, q2])


def recursive():
    recursive()


def test_recursive():
    get_referred_objects(recursive)


l = [q1, q2]


def refers_list():
    return len(l)


def test_refers_list():
    objs: List[Object] = get_referred_objects(refers_list)
    assert objs == []  # This may return [q1, q2] in the future


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


def test_nonglobal_function():
    def f():
        ...

    with pytest.raises(InvalidError, match=r"Cannot wrap `test_nonglobal_function.<locals>.f"):
        FunctionInfo(f)


class Foo:
    def __init__(self):
        pass

    @method()
    def bar(self):
        return "hello"

    @web_endpoint()
    def web(self):
        pass


def test_serialized_function_for_class():
    # The serialized function for the "service function" itself
    # should be a dict of all the _PartialFunction modal methods of
    # the class, to be used within the container entrypoint
    info = FunctionInfo(None, cls=Foo, serialized=True)

    serialized_function = info.serialized_function()
    revived_function = deserialize(serialized_function, None)
    assert isinstance(revived_function, dict)
    assert revived_function.keys() == {"bar", "web"}
    revived_bar = revived_function["bar"]
    assert isinstance(revived_bar, _PartialFunction)
    revived_web = revived_function["web"]
    assert revived_web.webhook_config and revived_web.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION
    assert revived_bar.raw_f.__get__(Foo())() == "hello"
