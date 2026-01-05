# Copyright Modal Labs 2025
import pytest

import modal
from modal.exception import InvalidError
from modal.server import Server, _Server

# ============ Basic Server Registration ============

server_app = modal.App("server-test-app", include_source=False)


@server_app.server(port=8000, serialized=True)
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
        service_function = GPUServer._get_service_function()
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
        service_function = ScalingServer._get_service_function()
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
    with pytest.raises(InvalidError, match="cannot have"):
        app = modal.App("server-method-test", include_source=False)

        @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
        class ServerWithMethod:
            @modal.method()
            def some_method(self):
                pass


def test_server_requires_port():
    """Test that @app.server() requires a port parameter."""
    with pytest.raises(TypeError):
        app = modal.App("server-no-port", include_source=False)

        @app.server()  # type: ignore  # Missing port
        class NoPortServer:
            pass


# ============ Lifecycle Tests ============


def test_server_obj_enter_lifecycle():
    """Test that _ServerObj correctly runs @modal.enter methods."""
    from modal.server import _ServerObj

    class TestServer:
        def __init__(self):
            self.entered = False

        @modal.enter()
        def on_enter(self):
            self.entered = True

    mock_server = type("MockServer", (), {"_user_server_function": None})()
    server_obj = _ServerObj(mock_server, TestServer)

    server_obj._user_server_instance = TestServer()

    server_obj._enter()

    assert server_obj._has_entered is True


def test_server_obj_enter_runs_once():
    """Test that @modal.enter methods only run once."""
    from modal.server import _ServerObj

    call_count = 0

    class TestServer:
        @modal.enter()
        def on_enter(self):
            nonlocal call_count
            call_count += 1

    mock_server = type("MockServer", (), {"_user_server_function": None})()
    server_obj = _ServerObj(mock_server, TestServer)
    server_obj._user_server_instance = TestServer()

    server_obj._enter()
    server_obj._enter()
    server_obj._enter()

    assert call_count == 1


# ============ from_name Tests ============


def test_server_from_name(client, servicer):
    """Test Server.from_name() lookup."""
    my_server = Server.from_name("my-app", "MyServer")

    # Should be lazy - not hydrated yet
    assert not my_server.is_hydrated


# ============ ServerObj API Tests ============


def test_server_callable_returns_server_obj():
    """Test that calling a Server returns a ServerObj."""
    from modal.server import _Server, _ServerObj

    server = _Server._from_loader(
        lambda *args: None,
        rep="TestServer",
        deps=lambda: [],
        load_context_overrides=None,
    )
    server._user_server = type("TestClass", (), {})

    obj = server()

    assert isinstance(obj, _ServerObj)


# ============ HTTP Config Tests ============


def test_server_http_config_parameters(client, servicer):
    """Test that HTTP config parameters are passed correctly."""
    app = modal.App("server-http-config-test", include_source=False)

    @app.server(
        port=9000,
        serialized=True,
    )
    class HTTPConfigServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = HTTPConfigServer._get_service_function()
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert function_def.http_config.port == 9000


# ============ Integration Tests ============


def test_server_creates_class_object(client, servicer):
    """Test that deploying a server creates the expected objects."""
    app = modal.App("server-objects-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class ObjectsServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        app_id = app.app_id
        objects = servicer.app_objects[app_id]

        assert "ObjectsServer" in objects
        assert "ObjectsServer.*" in objects

        server_id = objects["ObjectsServer"]
        assert server_id.startswith("cs-") or server_id.startswith("sv-")


def test_server_http_config_set(client, servicer):
    """Test that servers have http_config set correctly."""
    app = modal.App("server-http-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class HTTPServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = HTTPServer._get_service_function()
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
    assert isinstance(ServerWithLifecycle, _Server)

    # But we can get the original user class
    user_cls = ServerWithLifecycle._get_user_server()

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

    assert isinstance(SimpleServer, _Server)

    user_cls = SimpleServer._get_user_server()

    # Can instantiate the user class
    instance = user_cls()

    # It's an instance of the original class
    assert type(instance).__name__ == "SimpleServer"


def test_server_get_method_names_returns_empty():
    app = modal.App("server-methods-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class MethodlessServer:
        @modal.enter()
        def start(self):
            pass

    # Servers should return empty list for _get_method_names
    # This is used by CLI/import code that iterates registered classes
    assert list(MethodlessServer._get_method_names()) == []


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
