# Copyright Modal Labs 2023

from typing import List

from modal import Queue
from modal._utils.function_utils import FunctionInfo, get_referred_objects, method_has_params
from modal.object import Object

q1 = Queue.new()
q2 = Queue.new()


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
