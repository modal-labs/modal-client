# Copyright Modal Labs 2025
"""ASGI app wrapper for Modal servers (Flash architecture).

This module provides markers for ASGI/WSGI app factory functions that can be
used with `@app.server()`. The server decorator handles the actual class generation.

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

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Marker attribute to identify ASGI app factories
ASGI_APP_MARKER = "_modal_asgi_app_on_flash"
WSGI_APP_MARKER = "_modal_wsgi_app_on_flash"


def asgi_app_on_flash(
    *,
    startup_timeout: float = 30.0,
) -> Callable[[F], F]:
    """Decorator that marks a function as an ASGI app factory for use with `@app.server()`.

    This decorator marks the function so that `@app.server()` knows to wrap it
    with uvicorn startup logic automatically.

    Args:
        startup_timeout: Maximum time to wait for the server to start accepting
            connections. Defaults to 30 seconds.

    Returns:
        The marked function (to be processed by `@app.server()`).

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
        The port is specified in the `@app.server()` decorator. The ASGI
        server will be bound to 0.0.0.0 on that port.
    """

    def decorator(asgi_app_factory: F) -> F:
        # Mark the function as an ASGI app factory
        setattr(asgi_app_factory, ASGI_APP_MARKER, True)
        setattr(asgi_app_factory, "_asgi_startup_timeout", startup_timeout)
        return asgi_app_factory

    return decorator


def wsgi_app_on_flash(
    *,
    startup_timeout: float = 30.0,
) -> Callable[[F], F]:
    """Decorator that marks a function as a WSGI app factory for use with `@app.server()`.

    Similar to `asgi_app_on_flash`, but for WSGI applications (like Flask, Django).
    The WSGI app is wrapped with an ASGI adapter before being served.

    Args:
        startup_timeout: Maximum time to wait for the server to start accepting
            connections. Defaults to 30 seconds.

    Returns:
        The marked function (to be processed by `@app.server()`).

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

    def decorator(wsgi_app_factory: F) -> F:
        # Mark the function as a WSGI app factory
        setattr(wsgi_app_factory, WSGI_APP_MARKER, True)
        setattr(wsgi_app_factory, "_wsgi_startup_timeout", startup_timeout)
        return wsgi_app_factory

    return decorator


def create_asgi_server_class(asgi_app_factory: Callable, port: int) -> type:
    """Create a server class that runs the ASGI app factory with uvicorn.

    This is called internally by @app.server() when it receives an ASGI app factory.
    """
    import threading

    from .._partial_function import _PartialFunction, _PartialFunctionFlags, _PartialFunctionParams

    class_name = asgi_app_factory.__name__

    def _start_asgi_server(self):
        """Start the ASGI server in a background thread."""
        import uvicorn

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

    def _stop_asgi_server(self):
        """Clean up when the container shuts down."""
        pass

    # Create PartialFunction objects for the lifecycle methods
    enter_pf = _PartialFunction(
        _start_asgi_server,
        _PartialFunctionFlags.ENTER_POST_SNAPSHOT,
        _PartialFunctionParams(),
    )
    exit_pf = _PartialFunction(
        _stop_asgi_server,
        _PartialFunctionFlags.EXIT,
        _PartialFunctionParams(),
    )

    # Create the class
    server_class = type(
        class_name,
        (),
        {
            "__doc__": asgi_app_factory.__doc__ or f"Auto-generated server class for {class_name}.",
            "_asgi_app_factory": asgi_app_factory,
            "_server_thread": None,
            "_asgi_app": None,
            "_start_asgi_server": enter_pf,
            "_stop_asgi_server": exit_pf,
        },
    )

    return server_class


def create_wsgi_server_class(wsgi_app_factory: Callable, port: int) -> type:
    """Create a server class that runs the WSGI app factory with uvicorn.

    This is called internally by @app.server() when it receives a WSGI app factory.
    """
    import threading

    from .._partial_function import _PartialFunction, _PartialFunctionFlags, _PartialFunctionParams

    class_name = wsgi_app_factory.__name__

    def _start_wsgi_server(self):
        """Start the WSGI server in a background thread."""
        import uvicorn

        from modal._vendor.a2wsgi_wsgi import WSGIMiddleware

        host = "0.0.0.0"

        # Create the WSGI app and wrap with ASGI adapter
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

    enter_pf = _PartialFunction(
        _start_wsgi_server,
        _PartialFunctionFlags.ENTER_POST_SNAPSHOT,
        _PartialFunctionParams(),
    )
    exit_pf = _PartialFunction(
        _stop_wsgi_server,
        _PartialFunctionFlags.EXIT,
        _PartialFunctionParams(),
    )

    server_class = type(
        class_name,
        (),
        {
            "__doc__": wsgi_app_factory.__doc__ or f"Auto-generated server class for {class_name}.",
            "_wsgi_app_factory": wsgi_app_factory,
            "_server_thread": None,
            "_start_wsgi_server": enter_pf,
            "_stop_wsgi_server": exit_pf,
        },
    )

    return server_class
