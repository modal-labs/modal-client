# Copyright Modal Labs 2025
from modal.app import App


def f(x):
    assert App._get_container_app()
    return 123
