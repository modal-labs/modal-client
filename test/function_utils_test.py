# Copyright Modal Labs 2023

from typing import List

from modal import Queue
from modal._function_utils import get_referred_objects
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
