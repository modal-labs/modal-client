# Copyright Modal Labs 2022
import pytest


def pytest_markdown_docs_globals():
    import math

    import modal

    return {
        "modal": modal,
        "app": modal.App.lookup("my-app", create_if_missing=True),
        "math": math,
        "__name__": "runtest",
        "web_endpoint": modal.web_endpoint,
        "asgi_app": modal.asgi_app,
        "wsgi_app": modal.wsgi_app,
        "__file__": "xyz.py",
    }


@pytest.fixture(autouse=True)
def disable_auto_mount(monkeypatch):
    monkeypatch.setenv("MODAL_AUTOMOUNT", "0")
    yield
