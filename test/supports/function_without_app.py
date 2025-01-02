# Copyright Modal Labs 2024
from modal.app import App


def f(x):
    assert App.container_app()
    return 123
