# Copyright Modal Labs 2025
import pytest
import re
import uuid

import modal
import modal.experimental
from modal._serialization import deserialize
from modal._server import _Server
from modal.exception import InvalidError, NotFoundError
from modal.server import Server
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
        assert urls == ["https://modal-labs--#urlserver.modal-us-east.modal.direct"]


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

        # Servers use "#ClassName" naming convention
        assert "#ObjectsServer" in objects

        server_id = objects["#ObjectsServer"]
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
