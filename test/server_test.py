# Copyright Modal Labs 2025
import pytest
import re
import uuid

import modal
import modal.experimental
from modal._serialization import deserialize
from modal.exception import InvalidError, NotFoundError
from modal.server import Server, _Server
from modal_proto import api_pb2

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


def test_server_with_gpu_and_autoscaler_settings(client, servicer):
    """Test that @app.server() accepts GPU configuration and autoscaler settings."""
    app = modal.App("server-gpu-test", include_source=False)

    @app.server(port=8000, min_containers=2, max_containers=10, proxy_regions=["us-east"], gpu="A10G", serialized=True)
    class GPUServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = GPUServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert function_def.resources.gpu_config.gpu_type == "A10G"

        settings = function_def.autoscaler_settings
        assert settings.min_containers == 2
        assert settings.max_containers == 10


# ============ Validation Tests ============
class ServerWithInit:
    def __init__(self, value: int):
        self.value = value


def test_server_rejects_custom_init():
    """Test that servers cannot have custom __init__ methods."""
    with pytest.raises(
        InvalidError,
        match="cannot have a custom __init__ method",
    ):
        _Server.validate_construction_mechanism(ServerWithInit)


class ServerWithDefaultInit:
    pass


def test_server_allows_default_init():
    """Test that servers with default __init__ are accepted."""
    _Server.validate_construction_mechanism(ServerWithDefaultInit)


def test_server_rejects_method_decorator():
    """Test that @modal.method() cannot be used on server classes."""
    with pytest.raises(
        InvalidError, match=re.escape("cannot have @method() decorated functions. Servers only expose HTTP endpoints.")
    ):
        app = modal.App("server-method-test", include_source=False)

        @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
        class ServerWithMethod:
            @modal.enter()
            def start(self):
                pass

            @modal.method()
            def some_method(self):
                pass


def test_server_rejects_empty_proxy_regions():
    """Test that @app.server() requires a non-empty proxy_regions parameter."""
    with pytest.raises(InvalidError, match="The `proxy_regions` argument must be non-empty."):
        app = modal.App("server-empty-proxy-regions-test", include_source=False)

        @app.server(port=8000, proxy_regions=[], serialized=True)
        class EmptyProxyRegionsServer:
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


def test_server_rejects_concurrent_decorator():
    """Test that @modal.concurrent() cannot be used on server classes."""
    with pytest.raises(InvalidError, match="cannot have @concurrent"):
        app = modal.App("server-concurrent-test", include_source=False)

        @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
        @modal.concurrent(max_inputs=10)  # type: ignore
        class ConcurrentServer:
            @modal.enter()
            def start(self):
                pass


def test_server_rejects_http_server_decorator():
    """Test that @modal.experimental.http_server() cannot be used on server classes."""
    with pytest.raises(InvalidError, match=r"cannot have @modal\.experimental\.http_server\(\)"):
        app = modal.App("server-http-server-test", include_source=False)

        @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
        @modal.experimental.http_server(port=9000, proxy_regions=["us-east"])  # type: ignore
        class HttpServerDecoratorServer:
            @modal.enter()
            def start(self):
                pass


def test_server_snap_without_enable_memory_snapshot():
    """Test that @modal.enter(snap=True) without enable_memory_snapshot=True fails."""
    with pytest.raises(InvalidError, match="enable_memory_snapshot=True"):
        app = modal.App("server-snap-test", include_source=False)

        @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
        class SnapServer:
            @modal.enter(snap=True)
            def pre_snapshot(self):
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


def test_server_enter_pre_snapshot_runs_before_post_snapshot():
    """Test that @modal.enter(snap=True) methods run before @modal.enter(snap=False) methods."""
    from modal.server import _Server

    ordering_check = 1
    pre_snapshot_called = 0
    post_snapshot_called = 0

    class TestServer:
        @modal.enter(snap=True)
        def on_pre_snapshot(self):
            nonlocal pre_snapshot_called
            nonlocal ordering_check
            ordering_check -= 1
            pre_snapshot_called += 1

        @modal.enter(snap=False)
        def on_post_snapshot(self):
            nonlocal ordering_check
            assert ordering_check == 0, "post_snapshot should run after pre_snapshot"
            ordering_check += 10
            nonlocal post_snapshot_called
            post_snapshot_called += 1

    server = _Server()
    server._user_cls = TestServer
    server._user_cls_instance = TestServer()

    server._enter()
    server._enter()
    server._enter()
    server._enter()

    assert pre_snapshot_called == 1
    assert post_snapshot_called == 1
    assert ordering_check == 10


def test_server_enter_pre_snapshot_runs_before_post_snapshot_async():
    """Test that @modal.enter(snap=True) methods run before @modal.enter(snap=False) methods."""

    pre_snapshot_called = 0
    post_snapshot_called = 0

    class TestServer:
        @modal.enter(snap=True)
        async def on_pre_snapshot(self):
            nonlocal pre_snapshot_called
            pre_snapshot_called += 1

        @modal.enter(snap=False)
        async def on_post_snapshot(self):
            nonlocal post_snapshot_called
            post_snapshot_called += 1


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


def test_server_handlers():
    """Test that _find_partial_methods_for_user_cls correctly identifies lifecycle handlers for servers."""
    from modal._partial_function import _find_partial_methods_for_user_cls, _PartialFunction, _PartialFunctionFlags

    class ServerWithHandlers:
        @modal.enter(snap=True)
        def my_memory_snapshot(self):
            pass

        @modal.enter()
        def my_enter(self):
            pass

        @modal.exit()
        def my_exit(self):
            pass

    pfs: dict[str, _PartialFunction]

    pfs = _find_partial_methods_for_user_cls(ServerWithHandlers, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT)
    assert list(pfs.keys()) == ["my_memory_snapshot"]

    pfs = _find_partial_methods_for_user_cls(ServerWithHandlers, _PartialFunctionFlags.ENTER_POST_SNAPSHOT)
    assert list(pfs.keys()) == ["my_enter"]

    pfs = _find_partial_methods_for_user_cls(ServerWithHandlers, _PartialFunctionFlags.EXIT)
    assert list(pfs.keys()) == ["my_exit"]


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
        assert function_def._experimental_group_size == 2


# ============ from_name Tests ============


def test_server_from_name(client, servicer):
    """Test Server.from_name() lookup."""
    my_server = Server.from_name("my-app", "MyServer")

    # Should be lazy - not hydrated yet
    assert not my_server._get_service_function().is_hydrated


def test_server_from_name_failed_lookup_error(client, servicer):
    """Test that Server.from_name() raises NotFoundError with helpful message."""
    with pytest.raises(NotFoundError, match="Lookup failed.*MyServer.*my-nonexistent-app"):
        Server.from_name("my-nonexistent-app", "MyServer", client=client).hydrate()


def test_server_from_name_with_environment(client, servicer):
    """Test that Server.from_name() with environment_name includes it in error message."""
    with pytest.raises(NotFoundError, match="some-env"):
        Server.from_name("my-nonexistent-app", "MyServer", environment_name="some-env", client=client).hydrate()


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
        urls = URLServer.get_urls()  # type: ignore[attr-defined]
        # URLs are generated by the mock servicer based on function name and proxy regions
        assert urls == ["https://modal-labs--urlserver.modal-us-east.modal.direct"]


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
        AutoscaleServer.update_autoscaler(min_containers=1, max_containers=5)  # type: ignore[attr-defined]
        function_id = AutoscaleServer._get_service_function().object_id  # type: ignore[attr-defined]

    assert servicer.app_functions[function_id].autoscaler_settings.min_containers == 1
    assert servicer.app_functions[function_id].autoscaler_settings.max_containers == 5


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
        assert list(function_def.http_config.proxy_regions) == ["us-east"]
        assert function_def.http_config.startup_timeout == 30
        assert function_def.http_config.exit_grace_period == 0
        assert function_def.http_config.h2_enabled is False


# ============ Resource Configuration Tests ============


def test_server_target_concurrency(client, servicer):
    """Test that target_concurrency parameter is passed correctly."""
    app = modal.App("server-target-concurrency-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], target_concurrency=50, serialized=True)
    class ConcurrencyServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = ConcurrencyServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert function_def.target_concurrent_inputs == 50


def test_server_with_volumes(client, servicer):
    """Test that servers can mount volumes."""
    app = modal.App("server-volumes-test", include_source=False)
    vol = modal.Volume.from_name("test-volume", create_if_missing=True)

    @app.server(port=8000, proxy_regions=["us-east"], volumes={"/data": vol}, serialized=True)
    class VolumeServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = VolumeServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert len(function_def.volume_mounts) == 1
        assert function_def.volume_mounts[0].mount_path == "/data"


def test_server_with_secrets(client, servicer):
    """Test that servers can use secrets."""
    app = modal.App("server-secrets-test", include_source=False)
    secret = modal.Secret.from_dict({"API_KEY": "test-key"})

    @app.server(port=8000, proxy_regions=["us-east"], secrets=[secret], serialized=True)
    class SecretServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = SecretServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert len(function_def.secret_ids) >= 1


def test_server_with_image(client, servicer):
    """Test that servers can use custom images."""
    app = modal.App("server-image-test", include_source=False)
    custom_image = modal.Image.debian_slim().pip_install("flask")

    @app.server(port=8000, proxy_regions=["us-east"], image=custom_image, serialized=True)
    class ImageServer:
        @modal.enter()
        def start(self):
            try:
                import flask  # noqa: F401
            except ImportError:
                raise RuntimeError("flask is not installed")

            pass

    with app.run(client=client):
        service_function = ImageServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        # Verify image is set (image_id should be present)
        assert function_def.image_id is not None


def test_server_with_memory_and_cpu(client, servicer):
    """Test that memory and cpu parameters are passed correctly."""
    app = modal.App("server-resources-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], memory=2048, cpu=4.0, serialized=True)
    class ResourceServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = ResourceServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert function_def.resources.memory_mb == 2048
        assert function_def.resources.milli_cpu == 4000


def test_server_multiple_proxy_regions(client, servicer):
    """Test that servers can use multiple proxy regions."""
    app = modal.App("server-multi-region-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east", "us-west", "eu-west"], serialized=True)
    class MultiRegionServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = MultiRegionServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert list(function_def.http_config.proxy_regions) == ["us-east", "us-west", "eu-west"]


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

        server_id = objects["ObjectsServer"]
        assert server_id.startswith("fu-")


def test_server_with_inheritance(client, servicer):
    """Test that a server class can inherit from a base class with @modal.enter() methods."""
    app = modal.App("server-inheritance-test", include_source=False)

    class BaseServer:
        @modal.enter()
        def base_enter(self):
            self.base_entered = True

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class DerivedServer(BaseServer):
        @modal.enter()
        def derived_enter(self):
            self.derived_entered = True

    with app.run(client=client):
        assert isinstance(DerivedServer, Server)

        # Verify the server was created
        service_function = DerivedServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        assert function_id.startswith("fu-")

    # Test that both enter methods are found
    from modal._partial_function import _find_partial_methods_for_user_cls, _PartialFunctionFlags

    user_cls = DerivedServer._get_user_cls()  # type: ignore[attr-defined]
    enter_methods = _find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.ENTER_POST_SNAPSHOT)
    assert "base_enter" in enter_methods
    assert "derived_enter" in enter_methods


def test_server_serialization_roundtrip(client, servicer):
    """Test that server class can be serialized and deserialized correctly."""
    app = modal.App("server-serialization-test", include_source=False)

    @app.server(port=8000, proxy_regions=["us-east"], serialized=True)
    class SerializedServer:
        @modal.enter()
        def start(self):
            self.started = True

    with app.run(client=client):
        service_function = SerializedServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        # Verify it was serialized
        assert function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED
        assert function_def.class_serialized

        # Deserialize and verify it works
        user_cls = deserialize(function_def.class_serialized, client)
        instance = user_cls()
        assert hasattr(instance, "start")
        instance.start()
        assert instance.started is True


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


# ============ E2E Container Tests ============


def _container_args_for_server(
    module_name: str,
    function_name: str,
    http_config: "api_pb2.HTTPConfig",
    serialized_params: bytes = b"",
):
    """Create container arguments for a server test."""
    app_layout = api_pb2.AppLayout(
        objects=[
            api_pb2.Object(object_id="im-1"),
            api_pb2.Object(
                object_id="fu-123",
                function_handle_metadata=api_pb2.FunctionHandleMetadata(
                    function_name=function_name,
                ),
            ),
        ],
        function_ids={function_name: "fu-123"},
    )

    function_def = api_pb2.Function(
        module_name=module_name,
        function_name=function_name,
        is_class=True,
        definition_type=api_pb2.Function.DEFINITION_TYPE_FILE,
        http_config=http_config,
    )

    return api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id="fu-123",
        app_id="ap-1",
        function_def=function_def,
        serialized_params=serialized_params,
        checkpoint_id=f"ch-{uuid.uuid4()}",
        app_layout=app_layout,
    )


# @skip_github_non_linux
# def test_server_e2e_lifecycle(servicer):
#     """
#     End-to-end test that @app.server() runs enter methods correctly in a container.

#     This test runs the container entrypoint in-process with a server class,
#     verifying that:
#     1. The @modal.enter() method is called
#     2. Flash RPCs (register/deregister) are called appropriately
#     """
#     # Clear any previous Flash RPC calls
#     servicer.flash_rpc_calls = []

#     # Create http_config to enable Flash/server functionality
#     http_config = api_pb2.HTTPConfig(
#         port=8002,
#         startup_timeout=5,
#         exit_grace_period=0,
#         h2_enabled=False,
#     )

#     container_args = _container_args_for_server(
#         "test.supports.functions",
#         "ServerWithEnter",
#         http_config=http_config,
#     )

#     # Provide empty inputs - servers serve HTTP, they don't process function inputs
#     servicer.container_inputs = []

#     # Set up environment for container execution
#     env = {
#         "MODAL_SERVER_URL": servicer.container_addr,
#         "MODAL_TASK_ID": "ta-123",
#         "MODAL_IS_REMOTE": "1",
#     }

#     # Drop the module from sys.modules to ensure clean import
#     module_name = "test.supports.functions"
#     if module_name in sys.modules:
#         sys.modules.pop(module_name)

#     # Reset _App tracking state between runs
#     _App._all_apps.clear()

#     with Client(servicer.container_addr, api_pb2.CLIENT_TYPE_CONTAINER, None) as client:
#         try:
#             with mock.patch.dict(os.environ, env):
#                 main(container_args, client)
#         except UserException:
#             # Handle gracefully
#             pass

#     # Verify Flash RPCs were called
#     assert "register" in servicer.flash_rpc_calls, (
#         f"FlashContainerRegister was not called. RPC calls: {servicer.flash_rpc_calls}"
#     )
#     assert "deregister" in servicer.flash_rpc_calls, (
#         f"FlashContainerDeregister was not called. RPC calls: {servicer.flash_rpc_calls}"
#     )
