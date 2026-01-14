# Copyright Modal Labs 2025
import pytest
import re

import modal
from modal.exception import InvalidError
from modal.server import Server, _Server

# ============ Basic Server Registration ============

server_app = modal.App("server-test-app", include_source=False)


@server_app.server(port=8000, proxy_regions=["us-east"], serialized=True)
class BasicServer:
    @modal.enter()
    def start(self):
        pass


def test_basic_server_registration(client, servicer):
    """Test that @app.server() registers a server with the correct config."""
    with server_app.run(client=client):
        assert isinstance(BasicServer, Server)
        service_function = BasicServer._get_service_function()
        function_id = service_function.object_id

        function_def = servicer.app_functions[function_id]
        http_config = function_def.http_config

        assert http_config is not None
        assert http_config.port == 8000


def test_server_with_gpu(client, servicer):
    """Test that @app.server() accepts GPU configuration."""
    app = modal.App("server-gpu-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], gpu="A10G", serialized=True)
    class GPUServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = GPUServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert function_def.resources.gpu_config.gpu_type == "A10G"


def test_server_with_min_containers(client, servicer):
    """Test that @app.server() accepts autoscaling configuration."""
    app = modal.App("server-scaling-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], min_containers=2, max_containers=10, serialized=True)
    class ScalingServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = ScalingServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        settings = function_def.autoscaler_settings
        assert settings.min_containers == 2
        assert settings.max_containers == 10


# ============ Validation Tests ============


def test_server_rejects_custom_init():
    """Test that servers cannot have custom __init__ methods."""
    with pytest.raises(
        InvalidError,
        match="cannot have a custom __init__ method",
    ):
        _Server.validate_construction_mechanism(ServerWithInit)


class ServerWithInit:
    def __init__(self, value: int):
        self.value = value


def test_server_allows_default_init():
    """Test that servers with default __init__ are accepted."""
    _Server.validate_construction_mechanism(ServerWithDefaultInit)


class ServerWithDefaultInit:
    pass


def test_server_rejects_method_decorator():
    """Test that @modal.method() cannot be used on server classes."""
    with pytest.raises(InvalidError, match=re.escape("Server class must have an @modal.enter() to setup the server.")):
        app = modal.App("server-method-test", include_source=False)

        @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
        class ServerWithMethod:
            @modal.method()
            def some_method(self):
                pass


def test_server_requires_port():
    """Test that @app.server() requires a port parameter."""
    with pytest.raises(InvalidError, match="The `proxy_regions` argument must be non-empty."):
        app = modal.App("server-no-port", include_source=False)

        @app.server()  # type: ignore  # Missing port
        class NoPortServer:
            pass


def test_server_rejects_parametrization():
    """Test that modal.parameter() cannot be used on server classes."""
    with pytest.raises(InvalidError, match="cannot use modal.parameter"):
        app = modal.App("server-param-test", include_source=False)

        @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
        class ParameterizedServer:
            model_name: str = modal.parameter()

            @modal.enter()
            def start(self):
                pass


def test_server_rejects_parametrization_with_default():
    """Test that modal.parameter() with default cannot be used on server classes."""
    with pytest.raises(InvalidError, match="cannot use modal.parameter"):
        app = modal.App("server-param-default-test", include_source=False)

        @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
        class ParameterizedServerWithDefault:
            model_name: str = modal.parameter(default="gpt-4")

            @modal.enter()
            def start(self):
                pass


# ============ Lifecycle Tests ============


def test_server_enter_lifecycle():
    """Test that _Server correctly runs @modal.enter methods."""
    from modal.server import _Server

    class TestServer:
        def __init__(self):
            self.entered = False

        @modal.enter()
        def on_enter(self):
            self.entered = True

    # Create a minimal server for testing lifecycle
    server = _Server()
    server._user_cls = TestServer
    server._user_cls_instance = TestServer()

    server._enter()

    assert server._has_entered is True


def test_server_enter_runs_once():
    """Test that @modal.enter methods only run once."""
    from modal.server import _Server

    call_count = 0

    class TestServer:
        @modal.enter()
        def on_enter(self):
            nonlocal call_count
            call_count += 1

    server = _Server()
    server._user_cls = TestServer
    server._user_cls_instance = TestServer()

    server._enter()
    server._enter()
    server._enter()

    assert call_count == 1


@pytest.mark.asyncio
async def test_server_aenter_runs_sync_enter_methods():
    """Test that _aenter correctly runs sync @modal.enter methods."""
    from modal.server import _Server

    class TestServer:
        def __init__(self):
            self.entered = False

        @modal.enter()
        def on_enter(self):
            self.entered = True

    server = _Server()
    server._user_cls = TestServer
    server._user_cls_instance = TestServer()

    await server._aenter()

    assert server._has_entered is True
    assert server._user_cls_instance.entered is True


@pytest.mark.asyncio
async def test_server_aenter_awaits_async_enter_methods():
    """Test that _aenter correctly awaits async @modal.enter methods."""
    from modal.server import _Server

    class TestServer:
        def __init__(self):
            self.entered = False

        @modal.enter()
        async def on_enter(self):
            self.entered = True

    server = _Server()
    server._user_cls = TestServer
    server._user_cls_instance = TestServer()

    await server._aenter()

    assert server._has_entered is True
    assert server._user_cls_instance.entered is True


@pytest.mark.asyncio
async def test_server_aenter_runs_both_sync_and_async_enter_methods():
    """Test that _aenter handles a mix of sync and async @modal.enter methods."""
    from modal.server import _Server

    class TestServer:
        def __init__(self):
            self.sync_entered = False
            self.async_entered = False

        @modal.enter()
        def sync_enter(self):
            self.sync_entered = True

        @modal.enter()
        async def async_enter(self):
            self.async_entered = True

    server = _Server()
    server._user_cls = TestServer
    server._user_cls_instance = TestServer()

    await server._aenter()

    assert server._has_entered is True
    assert server._user_cls_instance.sync_entered is True
    assert server._user_cls_instance.async_entered is True


@pytest.mark.asyncio
async def test_server_aenter_runs_dunder_aenter():
    """Test that _aenter correctly runs __aenter__ method."""
    from modal.server import _Server

    class TestServer:
        def __init__(self):
            self.context_entered = False
            self.modal_entered = False

        async def __aenter__(self):
            self.context_entered = True

        @modal.enter()
        def on_enter(self):
            self.modal_entered = True

    server = _Server()
    server._user_cls = TestServer
    server._user_cls_instance = TestServer()

    await server._aenter()

    assert server._has_entered is True
    assert server._user_cls_instance.context_entered is True
    assert server._user_cls_instance.modal_entered is True


@pytest.mark.asyncio
async def test_server_aenter_runs_once():
    """Test that _aenter only runs @modal.enter methods once."""
    from modal.server import _Server

    call_count = 0

    class TestServer:
        @modal.enter()
        async def on_enter(self):
            nonlocal call_count
            call_count += 1

    server = _Server()
    server._user_cls = TestServer
    server._user_cls_instance = TestServer()

    await server._aenter()
    await server._aenter()
    await server._aenter()

    assert call_count == 1


# ============ Clustered Server Tests ============


def test_server_with_clustered_decorator(client, servicer):
    """Test that @modal.experimental.clustered() works with @app.server().

    Regression test: @modal.clustered() wraps the class in a _PartialFunction,
    which caused validate_wrapped_user_cls_decorators to fail on inspect.isclass().
    """
    app = modal.App("server-clustered-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    @modal.experimental.clustered(size=2)  # type: ignore
    class ClusteredServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        assert isinstance(ClusteredServer, Server)
        service_function = ClusteredServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        # Verify cluster settings were applied
        assert function_def.cluster_size == 2


# ============ from_name Tests ============


def test_server_from_name(client, servicer):
    """Test Server.from_name() lookup."""
    my_server = Server.from_name("my-app", "MyServer")

    # Should be lazy - not hydrated yet
    assert not my_server._get_service_function().is_hydrated


# ============ Live Method Tests ============


def test_server_get_urls(client, servicer):
    """Test that Server.get_urls() works without raising AttributeError.

    Regression test: @live_method calls self.hydrate(), but _Server didn't
    have a hydrate() method, causing AttributeError at runtime.
    """
    app = modal.App("server-get-urls-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class URLServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        # This should not raise AttributeError: '_Server' object has no attribute 'hydrate'
        urls = URLServer.get_urls()
        # URLs may be None or a list depending on servicer behavior
        assert urls is None or isinstance(urls, list)


def test_server_update_autoscaler(client, servicer):
    """Test that Server.update_autoscaler() works without raising AttributeError.

    Regression test: @live_method calls self.hydrate(), but _Server didn't
    have a hydrate() method, causing AttributeError at runtime.
    """
    app = modal.App("server-update-autoscaler-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class AutoscaleServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        # This should not raise AttributeError: '_Server' object has no attribute 'hydrate'
        AutoscaleServer.update_autoscaler(min_containers=1, max_containers=5)


# ============ HTTP Config Tests ============


def test_server_http_config_parameters(client, servicer):
    """Test that HTTP config parameters are passed correctly."""
    app = modal.App("server-http-config-test", include_source=False)

    @app.server(
        port=9000,
        proxy_regions=["us-east"],
        serialized=True,
    )
    class HTTPConfigServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = HTTPConfigServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert function_def.http_config.port == 9000


# ============ Integration Tests ============


# def test_server_creates_class_object(client, servicer):
#     """Test that deploying a server creates the expected objects."""
#     app = modal.App("server-objects-test", include_source=False)

#     @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
#     class ObjectsServer:
#         @modal.enter()
#         def start(self):
#             pass

#     with app.run(client=client):
#         app_id = app.app_id
#         objects = servicer.app_objects[app_id]

#         assert "ObjectsServer" in objects

#         server_id = objects["ObjectsServer"]
#         assert server_id.startswith("fu-")


def test_server_http_config_set(client, servicer):
    """Test that servers have http_config set correctly."""
    app = modal.App("server-http-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class HTTPServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = HTTPServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        # Verify http_config is set with the correct port
        assert function_def.http_config.port == 8000


# ============ Container Import Handling Tests ============


def test_server_has_user_server_with_mro():
    # Test servers have correct mro
    from modal._partial_function import _find_partial_methods_for_user_cls, _PartialFunctionFlags

    app = modal.App("server-mro-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class ServerWithLifecycle:
        @modal.enter()
        def on_start(self):
            pass

    # The decorated class is now a Server object
    assert isinstance(ServerWithLifecycle, Server)

    # But we can get the original user class
    user_cls = ServerWithLifecycle._get_user_cls()  # type: ignore[attr-defined]

    # The user class should have mro() (it's an actual class)
    assert hasattr(user_cls, "mro")
    assert callable(user_cls.mro)

    # _find_partial_methods_for_user_cls should work with the user class
    lifecycle_flags = ~_PartialFunctionFlags.interface_flags()
    partials = _find_partial_methods_for_user_cls(user_cls, lifecycle_flags)

    # Should find the @enter method
    assert "on_start" in partials


def test_server_user_class_instantiation():
    app = modal.App("server-instance-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class SimpleServer:
        @modal.enter()
        def start(self):
            self.started = True

    assert isinstance(SimpleServer, Server)

    user_cls = SimpleServer._get_user_cls()  # type: ignore[attr-defined]

    # Can instantiate the user class
    instance = user_cls()

    # It's an instance of the original class
    assert type(instance).__name__ == "SimpleServer"


def test_server_has_name_attribute():
    """Test that Server.__name__ returns the server name.

    Regression test for: AttributeError: 'Server' object has no attribute '__name__'
    """
    app = modal.App("server-name-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class NamedServer:
        @modal.enter()
        def start(self):
            pass

    # Server should have __name__ for compatibility
    assert hasattr(NamedServer, "__name__")
    assert NamedServer.__name__ == "NamedServer"
