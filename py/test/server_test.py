# Copyright Modal Labs 2025
import contextlib
import pytest
import re
import subprocess
from unittest import mock

import modal
import modal.experimental
from modal._serialization import deserialize
from modal._server import _Server
from modal.exception import InvalidError, NotFoundError
from modal.server import Server
from modal_proto import api_pb2
from test import conftest as client_test_conftest

# =============================================================================
# Basic Server Registration
# =============================================================================


@pytest.mark.asyncio
async def test_servicer_factory_uses_kernel_selected_tcp_ports(blob_server, credentials, monkeypatch):
    def fail_find_free_port() -> int:
        raise AssertionError("servicer_factory should not preselect TCP server ports")

    monkeypatch.setattr(client_test_conftest, "find_free_port", fail_find_free_port, raising=False)
    async with client_test_conftest.servicer_factory(blob_server, credentials) as servicer:
        assert not servicer.client_addr.endswith(":0")
        assert not servicer.task_command_router_url.endswith(":0")


server_app = modal.App("server-test-app", include_source=False)


@server_app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
class BasicServer:
    @modal.enter()
    def start(self):
        pass


def test_basic_server_registration(client, servicer):
    """Test that @app._experimental_server() registers a server with the correct config."""
    with server_app.run(client=client):
        assert isinstance(BasicServer, Server)
        service_function = BasicServer._get_service_function()
        function_id = service_function.object_id

        function_def = servicer.app_functions[function_id]
        http_config = function_def.http_config

        assert http_config is not None
        assert http_config.port == 8000


def test_server_with_gpu_and_autoscaler_settings(client, servicer):
    """Test that @app._experimental_server() accepts GPU configuration and autoscaler settings."""
    app = modal.App("server-gpu-test", include_source=False)

    @app._experimental_server(
        port=8000, min_containers=2, max_containers=10, routing_regions=["us-east"], gpu="A10G", serialized=True
    )
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


flash_app_default = modal.App("flash-app-default")


@flash_app_default._experimental_server(
    port=8080,
    routing_regions=["us-east", "us-west"],
    exit_grace_period=10,
    target_concurrency=10,
)
class FlashClassDefault:
    @modal.enter()
    def serve(self):
        self.process = subprocess.Popen(["python3", "-m", "http.server", "8080"])


app = modal.App("flash-app-2")


@app._experimental_server(
    port=8080,
    routing_regions=["us-east", "us-west", "ap-south"],
    startup_timeout=10,
    exit_grace_period=10,
    min_containers=1,
    image=modal.Image.debian_slim().pip_install("fastapi", "uvicorn"),
    target_concurrency=100,
)
class FlashClass:
    @modal.enter()
    def start(self):
        self.process = subprocess.Popen(["python3", "-m", "http.server", "8080"])


def test_run_server(client, servicer):
    """Test running server params are set correctly."""
    assert len(servicer.precreated_functions) == 0
    assert servicer.n_functions == 0
    with flash_app_default.run(client=client):
        service_function = FlashClassDefault._get_service_function()  # type: ignore
        method_handle_object_id = service_function.object_id
        assert isinstance(FlashClassDefault, Server)
        app_id = flash_app_default.app_id

    # Servers don't have class_ids, only service functions
    assert len(servicer.classes) == 0
    assert servicer.n_functions == 1
    objects = servicer.app_objects[app_id]
    # Servers use just the class name, not "ClassName.*"
    server_function_id = objects["FlashClassDefault"]
    assert servicer.precreated_functions == {server_function_id}
    assert method_handle_object_id == server_function_id
    assert len(objects) == 1  # just the service function
    assert server_function_id.startswith("fu-")
    assert servicer.app_functions[server_function_id].is_class

    assert servicer.app_functions[server_function_id].module_name == "test.server_test"
    assert servicer.app_functions[server_function_id].function_name == "FlashClassDefault"
    assert servicer.app_functions[server_function_id].target_concurrent_inputs == 10
    assert servicer.app_functions[server_function_id].method_definitions_set
    assert servicer.app_functions[server_function_id].startup_timeout_secs == 30
    assert servicer.app_functions[server_function_id].app_name == "flash-app-default"
    assert servicer.app_functions[server_function_id]._experimental_concurrent_cancellations


def test_run_server_normalizes_empty_checkpoint_id(client):
    import modal._container_entrypoint as container_entrypoint

    container_args = api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id="fu-123",
        function_def=api_pb2.Function(),
    )
    task_lifecycle_manager = mock.Mock()
    service = mock.MagicMock()
    event_loop = object()

    with (
        mock.patch.object(
            container_entrypoint,
            "TaskLifecycleManager",
            return_value=task_lifecycle_manager,
        ) as task_lifecycle_manager_cls,
        mock.patch.object(container_entrypoint, "hydrate_function", return_value=service),
        mock.patch.object(container_entrypoint, "call_server"),
        mock.patch.object(container_entrypoint, "UserCodeEventLoop") as user_code_event_loop_cls,
    ):
        user_code_event_loop_cls.return_value.__enter__.return_value = event_loop
        container_entrypoint.run_server(container_args, client)

    task_lifecycle_manager_cls.assert_called_once_with(
        container_args.task_id,
        container_args.function_id,
        container_args.function_def,
        None,
        client,
    )
    service.lifecycle_context.assert_called_once_with(event_loop, task_lifecycle_manager=task_lifecycle_manager)


def test_run_server_operation_order(client):
    import modal._container_entrypoint as container_entrypoint

    events = []
    function_def = api_pb2.Function(is_server=True, function_name="OrderedServer")
    container_args = api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id="fu-123",
        function_def=function_def,
    )
    task_lifecycle_manager = object()
    event_loop = object()
    service = mock.MagicMock()

    def task_lifecycle_manager_factory(*args):
        events.append("task_lifecycle_manager")
        assert args == (
            container_args.task_id,
            container_args.function_id,
            container_args.function_def,
            None,
            client,
        )
        return task_lifecycle_manager

    def hydrate_function(*args):
        events.append("hydrate_function")
        assert args[0] is container_args
        assert args[1] is task_lifecycle_manager
        assert args[2] == container_args.function_def
        assert args[4] is client
        return service

    def user_code_event_loop_factory():
        events.append("user_code_event_loop")
        event_loop_context = mock.MagicMock()

        def enter_side_effect():
            events.append("event_loop_enter")
            return event_loop

        def exit_side_effect(*args):
            events.append("event_loop_exit")
            # __exit__ must return None or a boolean (not a value from append)
            return None

        event_loop_context.__enter__.side_effect = enter_side_effect
        event_loop_context.__exit__.side_effect = exit_side_effect
        return event_loop_context

    @contextlib.contextmanager
    def lifecycle_context(*args, **kwargs):
        events.append("lifecycle_context_enter")
        assert args == (event_loop,)
        assert kwargs == {"task_lifecycle_manager": task_lifecycle_manager}
        try:
            yield
        finally:
            events.append("lifecycle_context_exit")

    def call_server(loop):
        events.append("call_server")
        assert loop is event_loop

    service.lifecycle_context.side_effect = lifecycle_context

    with (
        mock.patch.object(
            container_entrypoint,
            "TaskLifecycleManager",
            side_effect=task_lifecycle_manager_factory,
        ),
        mock.patch.object(container_entrypoint, "hydrate_function", side_effect=hydrate_function),
        mock.patch.object(container_entrypoint, "UserCodeEventLoop", side_effect=user_code_event_loop_factory),
        mock.patch.object(container_entrypoint, "call_server", side_effect=call_server),
    ):
        container_entrypoint.run_server(container_args, client)

    assert events == [
        "task_lifecycle_manager",
        "hydrate_function",
        "user_code_event_loop",
        "event_loop_enter",
        "lifecycle_context_enter",
        "call_server",
        "lifecycle_context_exit",
        "event_loop_exit",
    ]


def test_server_lifecycle_context_operation_order(monkeypatch):
    import modal._runtime.user_code_imports as user_code_imports

    events = []
    volume_commit_calls = []

    class OrderedService(user_code_imports.Service):
        app = mock.MagicMock()
        function_def = api_pb2.Function(is_checkpointing_function=True)
        service_deps = None
        user_cls_instance = None

        def get_finalized_functions(self, fun_def, container_io_manager):
            raise AssertionError("server lifecycle test should not finalize functions")

        @contextlib.contextmanager
        def lifecycle_presnapshot(self, event_loop, task_lifecycle_manager):
            events.append("presnapshot_enter")
            try:
                yield
            finally:
                events.append("presnapshot_exit")

        @contextlib.contextmanager
        def lifecycle_postsnapshot(self, event_loop, task_lifecycle_manager):
            events.append("postsnapshot_enter")
            try:
                yield
            finally:
                events.append("postsnapshot_exit")

    @contextlib.contextmanager
    def snapshot_context_manager():
        events.append("snapshot_enter")
        try:
            yield
        finally:
            events.append("snapshot_exit")

    def after_snapshot():
        events.append("after_snapshot")

    def disable_signals():
        events.append("disable_signals")
        return "int_handler", "usr1_handler"

    def try_enable_signals(int_handler, usr1_handler):
        events.append("try_enable_signals")
        assert int_handler == "int_handler"
        assert usr1_handler == "usr1_handler"

    monkeypatch.setenv("MODAL_ENABLE_SNAP_RESTORE", "1")
    monkeypatch.setattr(user_code_imports, "disable_signals", disable_signals)
    monkeypatch.setattr(user_code_imports, "try_enable_signals", try_enable_signals)

    task_lifecycle_manager = mock.MagicMock()
    task_lifecycle_manager.memory_snapshot.side_effect = lambda: events.append("memory_snapshot")

    def volume_commit_side_effect(volume_ids):
        events.append("volume_commit")
        volume_commit_calls.append(volume_ids)

    task_lifecycle_manager.volume_commit.side_effect = volume_commit_side_effect

    with OrderedService().lifecycle_context(
        event_loop=mock.MagicMock(),
        task_lifecycle_manager=task_lifecycle_manager,
        snapshot_context_manager=snapshot_context_manager(),
        after_snapshot=after_snapshot,
    ):
        events.append("call_server")

    assert events == [
        "presnapshot_enter",
        "snapshot_enter",
        "memory_snapshot",
        "snapshot_exit",
        "after_snapshot",
        "postsnapshot_enter",
        "call_server",
        "disable_signals",
        "postsnapshot_exit",
        "presnapshot_exit",
        "volume_commit",
        "try_enable_signals",
    ]
    assert volume_commit_calls == [[]]


flash_params_override_app = modal.App("flash-params-override")


@flash_params_override_app._experimental_server(
    port=8080,
    routing_regions=["us-west"],
    target_concurrency=11,
    experimental_options={"flash": "us-east"},
)
class FlashParamsOverrideClass:
    @modal.enter()
    def serve(self):
        pass


def test_flash_params_override_experimental_options(client, servicer):
    """Test experimental options work with server decorator."""
    with flash_params_override_app.run(client=client):
        assert isinstance(FlashParamsOverrideClass, Server)
        app_id = flash_params_override_app.app_id

        objects = servicer.app_objects[app_id]
        server_function_id = objects["FlashParamsOverrideClass"]

        assert servicer.app_functions[server_function_id].target_concurrent_inputs == 11
        assert servicer.app_functions[server_function_id].experimental_options["flash"] == "us-east"


# =============================================================================
# Validation Tests
# =============================================================================


class ServerWithInit:
    def __init__(self, value: int):
        self.value = value


def test_server_rejects_custom_init():
    """Test that servers cannot have custom __init__ methods."""
    with pytest.raises(
        InvalidError,
        match="cannot have a custom __init__ method",
    ):
        _Server._validate_construction_mechanism(ServerWithInit)


class ServerWithDefaultInit:
    pass


def test_server_allows_default_init():
    """Test that servers with default __init__ are accepted."""
    _Server._validate_construction_mechanism(ServerWithDefaultInit)


def test_server_rejects_method_decorator():
    """Test that @modal.method() cannot be used on server classes."""
    with pytest.raises(
        InvalidError,
        match=re.escape("cannot have `@modal.method()` decorated functions. Servers only expose HTTP endpoints."),
    ):
        app = modal.App("server-method-test", include_source=False)

        @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
        class ServerWithMethod:
            @modal.enter()
            def start(self):
                pass

            @modal.method()
            def some_method(self):
                pass


def test_server_rejects_empty_routing_regions():
    """Test that @app._experimental_server() requires a non-empty routing_regions parameter."""
    with pytest.raises(InvalidError, match="The `routing_regions` argument must be non-empty."):
        app = modal.App("server-empty-proxy-regions-test", include_source=False)

        @app._experimental_server(port=8000, routing_regions=[], serialized=True)
        class EmptyProxyRegionsServer:
            pass


def test_server_rejects_parametrization():
    """Test that modal.parameter() cannot be used on server classes."""
    with pytest.raises(InvalidError, match="cannot use modal.parameter"):
        app = modal.App("server-param-test", include_source=False)

        @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
        class ParameterizedServer:
            model_name: str = modal.parameter()

            @modal.enter()
            def start(self):
                pass


def test_server_rejects_parametrization_with_default():
    """Test that modal.parameter() with default cannot be used on server classes."""
    with pytest.raises(InvalidError, match="cannot use modal.parameter"):
        app = modal.App("server-param-default-test", include_source=False)

        @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
        class ParameterizedServerWithDefault:
            model_name: str = modal.parameter(default="gpt-4")

            @modal.enter()
            def start(self):
                pass


def test_server_rejects_concurrent_decorator():
    """Test that @modal.concurrent() cannot be used on server classes."""
    with pytest.raises(
        InvalidError,
        match=r"Server class ConcurrentServer cannot be decorated with `@modal\.concurrent\(\)`. "
        r"Please use `target_concurrency` param instead\.",
    ):
        app = modal.App("server-concurrent-test", include_source=False)

        @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
        @modal.concurrent(max_inputs=10)  # type: ignore
        class ConcurrentServer:
            @modal.enter()
            def start(self):
                pass


def test_server_rejects_http_server_decorator():
    """Test that @modal.experimental.http_server() cannot be used on server classes."""
    with pytest.raises(InvalidError, match=r"cannot have @modal\.experimental\.http_server\(\)"):
        app = modal.App("server-http-server-test", include_source=False)

        @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
        @modal.experimental.http_server(port=9000, proxy_regions=["us-east"])  # type: ignore
        class HttpServerDecoratorServer:
            @modal.enter()
            def start(self):
                pass


def test_server_snap_without_enable_memory_snapshot():
    """Test that @modal.enter(snap=True) without enable_memory_snapshot=True fails."""
    with pytest.raises(InvalidError, match="enable_memory_snapshot=True"):
        app = modal.App("server-snap-test", include_source=False)

        @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
        class SnapServer:
            @modal.enter(snap=True)
            def pre_snapshot(self):
                pass


# ============ Clustered Server Tests ============


def test_server_with_clustered_decorator(client, servicer):
    """Test that @modal.experimental.clustered() works with @app._experimental_server().

    Regression test: @modal.clustered() wraps the class in a _PartialFunction,
    which caused validate_wrapped_user_cls_decorators to fail on inspect.isclass().
    """
    app = modal.App("server-clustered-test", include_source=False)

    @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
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


# =============================================================================
# from_name Tests
# =============================================================================


def test_server_from_name(client, servicer):
    server_app.deploy(client=client)
    my_server = Server.from_name("server-test-app", "BasicServer", client=client)
    assert not my_server._get_service_function().is_hydrated
    urls = my_server.get_urls()
    assert urls == {"us-east": "https://modal-labs--basicserver.modal-us-east.modal.direct"}


def test_server_from_name_failed_lookup_error(client, servicer):
    """Test that Server.from_name() raises NotFoundError with helpful message."""
    with pytest.raises(NotFoundError, match="Lookup failed.*MyServer.*my-nonexistent-app"):
        Server.from_name("my-nonexistent-app", "MyServer", client=client).hydrate()


def test_server_from_name_with_environment(client, servicer):
    """Test that Server.from_name() with environment_name includes it in error message."""
    with pytest.raises(NotFoundError, match="some-env"):
        Server.from_name("my-nonexistent-app", "MyServer", environment_name="some-env", client=client).hydrate()


# =============================================================================
# Live Method Tests
# =============================================================================


def test_server_get_urls(client, servicer):
    """Test that Server.get_urls() works without raising AttributeError.

    Regression test: @live_method calls self.hydrate(), but _Server didn't
    have a hydrate() method, causing AttributeError at runtime.
    """
    app = modal.App("server-get-urls-test", include_source=False)

    @app._experimental_server(port=8000, routing_regions=["us-east", "us-west", "ap-south"], serialized=True)
    class URLServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        # This should not raise AttributeError: '_Server' object has no attribute 'hydrate'
        urls = URLServer.get_urls()  # type: ignore[attr-defined]
        # URLs are generated by the mock servicer based on function name and proxy regions
        assert urls == {
            "us-east": "https://modal-labs--urlserver.modal-us-east.modal.direct",
            "us-west": "https://modal-labs--urlserver.modal-us-west.modal.direct",
            "ap-south": "https://modal-labs--urlserver.modal-ap-south.modal.direct",
        }


def test_server_update_autoscaler(client, servicer):
    """Test that Server.update_autoscaler() works without raising AttributeError.

    Regression test: @live_method calls self.hydrate(), but _Server didn't
    have a hydrate() method, causing AttributeError at runtime.
    """
    app = modal.App("server-update-autoscaler-test", include_source=False)

    @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
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


# =============================================================================
# HTTP Config Tests
# =============================================================================


def test_server_http_config_parameters(client, servicer):
    """Test that HTTP config parameters are passed correctly."""
    app = modal.App("server-http-config-test", include_source=False)

    @app._experimental_server(
        port=9000,
        routing_regions=["us-east"],
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


# =============================================================================
# Resource Configuration Tests
# =============================================================================


def test_server_target_concurrency(client, servicer):
    """Test that target_concurrency parameter is passed correctly."""
    app = modal.App("server-target-concurrency-test", include_source=False)

    @app._experimental_server(port=8000, routing_regions=["us-east"], target_concurrency=50, serialized=True)
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

    @app._experimental_server(port=8000, routing_regions=["us-east"], volumes={"/data": vol}, serialized=True)
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

    @app._experimental_server(port=8000, routing_regions=["us-east"], secrets=[secret], serialized=True)
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

    @app._experimental_server(port=8000, routing_regions=["us-east"], image=custom_image, serialized=True)
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

    @app._experimental_server(port=8000, routing_regions=["us-east"], memory=2048, cpu=4.0, serialized=True)
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


def test_server_multiple_routing_regions(client, servicer):
    """Test that servers can use multiple proxy regions."""
    app = modal.App("server-multi-region-test", include_source=False)

    @app._experimental_server(port=8000, routing_regions=["us-east", "us-west", "eu-west"], serialized=True)
    class MultiRegionServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = MultiRegionServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert list(function_def.http_config.proxy_regions) == ["us-east", "us-west", "eu-west"]


# =============================================================================
# Integration Tests
# =============================================================================


def test_server_creates_class_object(client, servicer):
    """Test that deploying a server creates the expected objects."""
    app = modal.App("server-objects-test", include_source=False)

    @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
    class ObjectsServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        app_id = app.app_id
        objects = servicer.app_objects[app_id]

        # Servers use "#ClassName" naming convention
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

    @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
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

    @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
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


# =============================================================================
# Container Import Handling Tests
# =============================================================================


def test_server_has_user_server_with_mro():
    # Test servers have correct mro
    from modal._partial_function import _find_partial_methods_for_user_cls, _PartialFunctionFlags

    app = modal.App("server-mro-test", include_source=False)

    @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
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

    @app._experimental_server(port=8000, routing_regions=["us-east"], serialized=True)
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
