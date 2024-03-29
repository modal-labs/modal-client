# Copyright Modal Labs 2022
def pytest_markdown_docs_globals():
    import math

    import modal

    return {
        "modal": modal,
        "stub": modal.Stub(),
        "math": math,
        "__name__": "runtest",
        "web_endpoint": modal.web_endpoint,
        "asgi_app": modal.asgi_app,
        "wsgi_app": modal.wsgi_app,
        "__file__": "xyz.py",
    }
