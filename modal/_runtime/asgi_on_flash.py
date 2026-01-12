# Copyright Modal Labs 2025

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Marker attribute to identify ASGI app factories
ASGI_APP_MARKER = "_modal_asgi_app_on_flash"
WSGI_APP_MARKER = "_modal_wsgi_app_on_flash"


def asgi_app_on_flash(
    *,
    startup_timeout: float = 30.0,
) -> Callable[[F], F]:
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
    module_name = getattr(asgi_app_factory, "__module__", __name__)

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

    server_class = type(
        class_name,
        (),
        {
            "__doc__": asgi_app_factory.__doc__ or f"Auto-generated server class for {class_name}.",
            "__module__": module_name,
            "__qualname__": class_name,
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
    module_name = getattr(wsgi_app_factory, "__module__", __name__)

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
            "__module__": module_name,
            "__qualname__": class_name,
            "_wsgi_app_factory": wsgi_app_factory,
            "_server_thread": None,
            "_start_wsgi_server": enter_pf,
            "_stop_wsgi_server": exit_pf,
        },
    )

    return server_class
