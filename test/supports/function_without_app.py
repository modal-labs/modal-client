# Copyright Modal Labs 2024
from modal.app import _App


def f(x):
    assert _App._container_app
    return 123
