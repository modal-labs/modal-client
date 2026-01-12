# Copyright Modal Labs 2025
"""ASGI app wrapper for Modal servers (Flash architecture).

This module provides the `asgi_app_on_flash` decorator that wraps an ASGI application
factory function into a class suitable for use with `@app.server()`.

Example:
    @app.server(port=8000, proxy_regions=["us-east"], image=image)
    @modal.experimental.asgi_app_on_flash()
    def create_app():
        from fastapi import FastAPI
        app = FastAPI()

        @app.get("/")
        def root():
            return {"message": "Hello World"}

        return app
"""

import threading
from typing import Any, Callable, Optional, TypeVar

from .._partial_function import (
    _PartialFunction,
    _PartialFunctionFlags,
    _PartialFunctionParams,
)
from .._utils.async_utils import synchronizer

F = TypeVar("F", bound=Callable[..., Any])

# Default timeout for waiting for the web server to start
DEFAULT_STARTUP_TIMEOUT = 30.0


def _make_enter_partial(method: Callable) -> Any:
    """Create a PartialFunction for an @enter method and wrap it for synchronicity."""
    flags = _PartialFunctionFlags.ENTER_POST_SNAPSHOT
    params = _PartialFunctionParams()
    pf = _PartialFunction(method, flags, params)
    # Wrap with synchronicity so it's recognized as PartialFunction by validation
    return synchronizer._translate_out(pf, interface=None)


def _make_exit_partial(method: Callable) -> Any:
    """Create a PartialFunction for an @exit method and wrap it for synchronicity."""
    flags = _PartialFunctionFlags.EXIT
    params = _PartialFunctionParams()
    pf = _PartialFunction(method, flags, params)
    # Wrap with synchronicity so it's recognized as PartialFunction by validation
    return synchronizer._translate_out(pf, interface=None)


def asgi_app_on_flash(
    *,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    label: Optional[str] = None,
) -> Callable[[F], type]:
    """Decorator that wraps an ASGI app factory into a class for `@app.server()`.

    This decorator takes a function that returns an ASGI application (like FastAPI,
    Starlette, etc.) and creates a class with an `@modal.enter()` method that
    starts the server using uvicorn.

    Args:
        startup_timeout: Maximum time to wait for the server to start accepting
            connections. Defaults to 30 seconds.
        label: Optional label for the generated class. Defaults to the function name.

    Returns:
        A class that can be used with `@app.server()`.

    Example:
        ```python
        @app.server(port=8000, proxy_regions=["us-east"], image=image)
        @modal.experimental.asgi_app_on_flash()
        def my_fastapi_app():
            from fastapi import FastAPI
            app = FastAPI()

            @app.get("/")
            def root():
                return {"hello": "world"}

            return app
        ```

    Note:
        The port must be specified in the `@app.server()` decorator, and the ASGI
        server will be bound to 0.0.0.0 on that port.
    """

    def decorator(asgi_app_factory: F) -> type:
        # Get the function name for the class name
        class_name = label or asgi_app_factory.__name__

        # Define the enter method
        def _start_asgi_server(self):
            """Start the ASGI server in a background thread."""
            import uvicorn

            # Get the port from the server's service function
            # This is injected by @app.server() via _flash_port attribute
            port = getattr(self.__class__, "_flash_port", 8000)
            host = "0.0.0.0"

            # Create the ASGI app by calling the factory
            self._asgi_app = asgi_app_factory()

            # Configure uvicorn
            config = uvicorn.Config(
                self._asgi_app,
                host=host,
                port=port,
                log_level="info",
            )
            server = uvicorn.Server(config)

            # Start uvicorn in a daemon thread
            self._server_thread = threading.Thread(
                target=server.run,
                daemon=True,
            )
            self._server_thread.start()

        # Define the exit method
        def _stop_asgi_server(self):
            """Clean up when the container shuts down."""
            # The daemon thread will be terminated automatically when the process exits
            pass

        # Create the class dynamically with PartialFunction attributes
        # Using type() ensures the methods are properly set as class attributes
        ASGIServerClass = type(
            class_name,
            (),
            {
                "__doc__": asgi_app_factory.__doc__ or "Auto-generated server class for ASGI app.",
                "_asgi_app_factory": asgi_app_factory,
                "_startup_timeout": startup_timeout,
                "_server_thread": None,
                "_asgi_app": None,
                "_start_asgi_server": _make_enter_partial(_start_asgi_server),
                "_stop_asgi_server": _make_exit_partial(_stop_asgi_server),
            },
        )

        return ASGIServerClass

    return decorator


def wsgi_app_on_flash(
    *,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    label: Optional[str] = None,
) -> Callable[[F], type]:
    """Decorator that wraps a WSGI app factory into a class for `@app.server()`.

    Similar to `asgi_app_on_flash`, but for WSGI applications (like Flask, Django).
    The WSGI app is wrapped with an ASGI adapter before being served.

    Args:
        startup_timeout: Maximum time to wait for the server to start accepting
            connections. Defaults to 30 seconds.
        label: Optional label for the generated class. Defaults to the function name.

    Returns:
        A class that can be used with `@app.server()`.

    Example:
        ```python
        @app.server(port=8000, proxy_regions=["us-east"], image=image)
        @modal.experimental.wsgi_app_on_flash()
        def my_flask_app():
            from flask import Flask
            app = Flask(__name__)

            @app.route("/")
            def root():
                return {"hello": "world"}

            return app
        ```
    """

    def decorator(wsgi_app_factory: F) -> type:
        class_name = label or wsgi_app_factory.__name__

        def _start_wsgi_server(self):
            """Start the WSGI server in a background thread using uvicorn with ASGI adapter."""
            import uvicorn

            from modal._vendor.a2wsgi_wsgi import WSGIMiddleware

            port = getattr(self.__class__, "_flash_port", 8000)
            host = "0.0.0.0"

            # Create the WSGI app and wrap it with ASGI adapter
            wsgi_app = wsgi_app_factory()
            asgi_app = WSGIMiddleware(wsgi_app)

            config = uvicorn.Config(
                asgi_app,
                host=host,
                port=port,
                log_level="info",
            )
            server = uvicorn.Server(config)

            self._server_thread = threading.Thread(
                target=server.run,
                daemon=True,
            )
            self._server_thread.start()

        def _stop_wsgi_server(self):
            """Clean up when the container shuts down."""
            pass

        WSGIServerClass = type(
            class_name,
            (),
            {
                "__doc__": wsgi_app_factory.__doc__ or "Auto-generated server class for WSGI app.",
                "_wsgi_app_factory": wsgi_app_factory,
                "_startup_timeout": startup_timeout,
                "_server_thread": None,
                "_start_wsgi_server": _make_enter_partial(_start_wsgi_server),
                "_stop_wsgi_server": _make_exit_partial(_stop_wsgi_server),
            },
        )

        return WSGIServerClass

    return decorator
