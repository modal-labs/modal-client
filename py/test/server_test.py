# Copyright Modal Labs 2025
import contextlib
import pytest
import re
import subprocess
from typing import Any, cast
from unittest import mock

import modal
import modal.experimental
from modal._serialization import deserialize
from modal._server import _Server
from modal._utils.async_utils import synchronizer
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


@server_app.server(port=8000, routing_region="us-east", serialized=True)
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


def test_server_object_id_matches_service_function(client, servicer):
    """Test that Server.object_id returns the underlying service function's object_id."""
    with server_app.run(client=client):
        assert isinstance(BasicServer, Server)
        service_function = BasicServer._get_service_function()
        assert BasicServer.object_id == service_function.object_id
        assert BasicServer.object_id.startswith("fu-")


def test_server_with_gpu_and_autoscaler_settings(client, servicer):
    """Test that @app.server() accepts GPU configuration and autoscaler settings."""
    app = modal.App("server-gpu-test", include_source=False)

    @app.server(port=8000, min_containers=2, max_containers=10, routing_region="us-east", gpu="A10G", serialized=True)
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


@flash_app_default.server(
    port=8080,
    routing_region="us-east",
    exit_grace_period=10,
    target_concurrency=10,
)
class FlashClassDefault:
    @modal.enter()
    def serve(self):
        self.process = subprocess.Popen(["python3", "-m", "http.server", "8080"])


app = modal.App("flash-app-2")


@app.server(
    port=8080,
    routing_region="us-east",
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

    _client = cast(modal.client._Client, synchronizer._translate_in(client))
    function_def = api_pb2.Function(is_server=True, function_name="OrderedServer")
    container_args = api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id="fu-123",
        function_def=function_def,
    )
    event_loop = object()

    class OrderedServer:
        def __init__(self):
            self.side_effects = ["__init__"]

        task_lifecycle_manager: "OrderedTaskLifecycleManager"

        @contextlib.contextmanager
        def lifecycle_context(self, *args, **kwargs):
            self.side_effects.append("lifecycle_context_enter")
            assert args == (event_loop,)
            assert kwargs == {"task_lifecycle_manager": self.task_lifecycle_manager}
            try:
                yield
            finally:
                self.side_effects.append("lifecycle_context_exit")

    ordered_server = OrderedServer()

    class OrderedTaskLifecycleManager:
        def __init__(self, *args):
            ordered_server.side_effects.append("task_lifecycle_manager")
            assert args == (
                container_args.task_id,
                container_args.function_id,
                container_args.function_def,
                None,
                client,
            )
            ordered_server.task_lifecycle_manager = self

    def hydrate_function(*args):
        ordered_server.side_effects.append("hydrate_function")
        assert args[0] is container_args
        assert args[1] is ordered_server.task_lifecycle_manager
        assert args[2] == container_args.function_def
        assert args[3] is _client
        return ordered_server

    class OrderedUserCodeEventLoop:
        def __init__(self):
            ordered_server.side_effects.append("user_code_event_loop")

        def __enter__(self):
            ordered_server.side_effects.append("event_loop_enter")
            return event_loop

        def __exit__(self, *args):
            ordered_server.side_effects.append("event_loop_exit")
            return None

    def call_server(loop):
        ordered_server.side_effects.append("call_server")
        assert loop is event_loop

    with (
        mock.patch.object(container_entrypoint, "TaskLifecycleManager", OrderedTaskLifecycleManager),
        mock.patch.object(container_entrypoint, "hydrate_function", hydrate_function),
        mock.patch.object(container_entrypoint, "UserCodeEventLoop", OrderedUserCodeEventLoop),
        mock.patch.object(container_entrypoint, "call_server", call_server),
    ):
        container_entrypoint.run_server(container_args, client)

    assert ordered_server.side_effects == [
        "__init__",
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

    class OrderedServer:
        def __init__(self):
            self.side_effects = ["__init__"]

        @modal.enter(snap=True)
        def presnap(self):
            self.side_effects.append("presnap")

        @modal.enter()
        def postsnap(self):
            self.side_effects.append("postsnap")

        @modal.exit()
        def on_exit(self):
            self.side_effects.append("exit")

    ordered_server = OrderedServer()
    volume_commit_calls = []

    @contextlib.contextmanager
    def snapshot_context_manager():
        ordered_server.side_effects.append("snapshot_enter")
        try:
            yield
        finally:
            ordered_server.side_effects.append("snapshot_exit")

    def after_snapshot():
        ordered_server.side_effects.append("after_snapshot")

    def disable_signals():
        ordered_server.side_effects.append("disable_signals")
        return "int_handler", "usr1_handler"

    def try_enable_signals(int_handler, usr1_handler):
        ordered_server.side_effects.append("try_enable_signals")
        assert int_handler == "int_handler"
        assert usr1_handler == "usr1_handler"

    monkeypatch.setenv("MODAL_ENABLE_SNAP_RESTORE", "1")
    monkeypatch.setattr(user_code_imports, "disable_signals", disable_signals)
    monkeypatch.setattr(user_code_imports, "try_enable_signals", try_enable_signals)

    class OrderedTaskLifecycleManager:
        @contextlib.contextmanager
        def handle_task_lifecycle_exception(self):
            yield

        def memory_snapshot(self):
            ordered_server.side_effects.append("memory_snapshot")

        def volume_commit(self, volume_ids):
            ordered_server.side_effects.append("volume_commit")
            volume_commit_calls.append(volume_ids)

    task_lifecycle_manager = OrderedTaskLifecycleManager()
    service = user_code_imports.ImportedServer(
        user_cls_instance=ordered_server,
        app=mock.MagicMock(),
        service_deps=None,
        function_def=api_pb2.Function(is_checkpointing_function=True),
    )

    with service.lifecycle_context(
        event_loop=mock.MagicMock(),
        task_lifecycle_manager=cast(Any, task_lifecycle_manager),
        snapshot_context_manager=snapshot_context_manager(),
        after_snapshot=after_snapshot,
    ):
        ordered_server.side_effects.append("call_server")

    assert ordered_server.side_effects == [
        "__init__",
        "presnap",
        "snapshot_enter",
        "memory_snapshot",
        "snapshot_exit",
        "after_snapshot",
        "postsnap",
        "call_server",
        "disable_signals",
        "exit",
        "volume_commit",
        "try_enable_signals",
    ]
    assert volume_commit_calls == [[]]


flash_params_override_app = modal.App("flash-params-override")


@flash_params_override_app.server(
    port=8080,
    routing_region="us-west",
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

        @app.server(port=8000, routing_region="us-east", serialized=True)
        class ServerWithMethod:
            @modal.enter()
            def start(self):
                pass

            @modal.method()
            def some_method(self):
                pass


def test_server_rejects_empty_routing_region():
    """Test that @app.server() requires a non-empty routing_region parameter."""
    with pytest.raises(InvalidError, match="The `routing_region` argument must be passed."):
        app = modal.App("server-empty-proxy-regions-test", include_source=False)

        @app.server(port=8000, routing_region="", serialized=True)
        class EmptyProxyRegionsServer:
            pass


def test_server_rejects_parametrization():
    """Test that modal.parameter() cannot be used on server classes."""
    with pytest.raises(InvalidError, match="cannot use modal.parameter"):
        app = modal.App("server-param-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", serialized=True)
        class ParameterizedServer:
            model_name: str = modal.parameter()

            @modal.enter()
            def start(self):
                pass


def test_server_rejects_parametrization_with_default():
    """Test that modal.parameter() with default cannot be used on server classes."""
    with pytest.raises(InvalidError, match="cannot use modal.parameter"):
        app = modal.App("server-param-default-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", serialized=True)
        class ParameterizedServerWithDefault:
            model_name: str = modal.parameter(default="gpt-4")

            @modal.enter()
            def start(self):
                pass


def test_server_rejects_parametrized_invocation():
    """Test that a server cannot be parametrized like a Cls (e.g. `MyServer(x=1)`).

    Servers only expose HTTP endpoints, so unlike `@app.cls()` classes they cannot
    be instantiated/parametrized at the call site.
    """
    app = modal.App("server-parametrized-call-test", include_source=False)

    @app.server(port=8000, routing_region="us-east", serialized=True)
    class ParametrizedCallServer:
        @modal.enter()
        def start(self):
            pass

    assert isinstance(ParametrizedCallServer, Server)
    with pytest.raises(TypeError, match="not callable"):
        ParametrizedCallServer(model_name="gpt-4")  # type: ignore[operator]


def test_server_rejects_concurrent_decorator():
    """Test that @modal.concurrent() cannot be used on server classes."""
    with pytest.raises(
        InvalidError,
        match=r"Server class ConcurrentServer cannot be decorated with `@modal\.concurrent\(\)`. "
        r"Please use `target_concurrency` param instead\.",
    ):
        app = modal.App("server-concurrent-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", serialized=True)
        @modal.concurrent(max_inputs=10)  # type: ignore
        class ConcurrentServer:
            @modal.enter()
            def start(self):
                pass


def test_server_rejects_http_server_decorator():
    """Test that @modal.experimental.http_server() cannot be used on server classes."""
    with pytest.raises(InvalidError, match=r"cannot have @modal\.experimental\.http_server\(\)"):
        app = modal.App("server-http-server-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", serialized=True)
        @modal.experimental.http_server(port=9000, proxy_regions=["us-east"])  # type: ignore
        class HttpServerDecoratorServer:
            @modal.enter()
            def start(self):
                pass


def test_server_rejects_batched_decorator():
    """Test that @modal.batched() cannot be stacked on server classes."""
    with pytest.raises(
        InvalidError,
        match=re.escape("Cannot apply `@modal.batched` to a class."),
    ):
        app = modal.App("server-batched-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", serialized=True)
        @modal.batched(max_batch_size=4, wait_ms=1000)  # type: ignore
        class BatchedServer:
            @modal.enter()
            def start(self):
                pass


def test_server_rejects_schedule():
    """Test that a schedule (via @app.function) cannot be stacked on server classes."""
    with pytest.raises(TypeError, match="cannot be used on a class"):
        app = modal.App("server-schedule-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", serialized=True)  # type: ignore[arg-type]
        @app.function(schedule=modal.Period(seconds=10))
        class ScheduledServer:
            @modal.enter()
            def start(self):
                pass


def test_server_rejects_cron():
    """Test that a cron job (via @app.function) cannot be stacked on server classes."""
    with pytest.raises(TypeError, match="cannot be used on a class"):
        app = modal.App("server-cron-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", serialized=True)  # type: ignore[arg-type]
        @app.function(schedule=modal.Cron("* * * * *"))
        class CronServer:
            @modal.enter()
            def start(self):
                pass


def test_server_snap_without_enable_memory_snapshot():
    """Test that @modal.enter(snap=True) without enable_memory_snapshot=True fails."""
    with pytest.raises(InvalidError, match="enable_memory_snapshot=True"):
        app = modal.App("server-snap-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", serialized=True)
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

    @app.server(port=8000, routing_region="us-east", serialized=True)
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


def test_server_with_proxy(client, servicer):
    """Test that @app.server() can be configured with a Modal Proxy."""
    app = modal.App("server-proxy-test", include_source=False)

    @app.server(
        port=8000,
        routing_region="us-east",
        proxy=modal.Proxy.from_name("my-proxy"),
        serialized=True,
    )
    class ProxyServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        assert isinstance(ProxyServer, Server)
        service_function = ProxyServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert function_def.proxy_id == "pr-123"


# =============================================================================
# from_name Tests
# =============================================================================


def test_server_from_name(client, servicer):
    server_app.deploy(client=client)
    my_server = Server.from_name("server-test-app", "BasicServer", client=client)
    assert not my_server._get_service_function().is_hydrated
    url = my_server.get_url()
    assert url == "https://modal-labs--basicserver.modal-us-east.modal.direct"


def test_server_from_name_object_id_matches_created(client, servicer):
    """Test that a Server resolved via from_name() has the same object_id as the deployed Server."""
    server_app.deploy(client=client)
    assert isinstance(BasicServer, Server)
    created_object_id = BasicServer.object_id

    my_server = Server.from_name("server-test-app", "BasicServer", client=client)
    my_server.hydrate(client=client)
    assert my_server.object_id == created_object_id


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


def test_server_get_url(client, servicer):
    """Test that Server.get_url() works without raising AttributeError.

    Regression test: @live_method calls self.hydrate(), but _Server didn't
    have a hydrate() method, causing AttributeError at runtime.
    """
    app = modal.App("server-get-urls-test", include_source=False)

    @app.server(port=8000, routing_region="us-east", serialized=True)
    class URLServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        # This should not raise AttributeError: '_Server' object has no attribute 'hydrate'
        url = URLServer.get_url()  # type: ignore[attr-defined]
        # URLs are generated by the mock servicer based on function name and proxy regions
        assert url == "https://modal-labs--urlserver.modal-us-east.modal.direct"


def test_server_update_autoscaler(client, servicer):
    """Test that Server.update_autoscaler() works without raising AttributeError.

    Regression test: @live_method calls self.hydrate(), but _Server didn't
    have a hydrate() method, causing AttributeError at runtime.
    """
    app = modal.App("server-update-autoscaler-test", include_source=False)

    @app.server(port=8000, routing_region="us-east", serialized=True)
    class AutoscaleServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        # This should not raise AttributeError: '_Server' object has no attribute 'hydrate'
        function_id = AutoscaleServer._get_service_function().object_id  # type: ignore[attr-defined]
        assert servicer.app_functions[function_id].target_concurrent_inputs == 0

        update_autoscaler = AutoscaleServer.update_autoscaler  # type: ignore[attr-defined]
        update_autoscaler(min_containers=1, max_containers=5, target_concurrency=20)  # type: ignore[call-arg]
        assert servicer.app_functions[function_id].autoscaler_settings.target_concurrency == 20
        assert servicer.app_functions[function_id].target_concurrent_inputs == 20

        update_autoscaler(target_concurrency=0)  # type: ignore[call-arg]

    assert servicer.app_functions[function_id].autoscaler_settings.min_containers == 1
    assert servicer.app_functions[function_id].autoscaler_settings.max_containers == 5
    assert servicer.app_functions[function_id].autoscaler_settings.HasField("target_concurrency")
    assert servicer.app_functions[function_id].autoscaler_settings.target_concurrency == 0
    assert servicer.app_functions[function_id].target_concurrent_inputs == 0


# =============================================================================
# HTTP Config Tests
# =============================================================================


def test_server_http_config_parameters(client, servicer):
    """Test that HTTP config parameters are passed correctly."""
    app = modal.App("server-http-config-test", include_source=False)

    @app.server(
        port=9000,
        routing_region="us-east",
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

    @app.server(port=8000, routing_region="us-east", target_concurrency=50, serialized=True)
    class ConcurrencyServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = ConcurrencyServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert function_def.target_concurrent_inputs == 50


def test_server_target_concurrency_zero(client, servicer):
    """Test that target_concurrency=0 disables the server target."""
    app = modal.App("server-zero-target-concurrency-test", include_source=False)

    @app.server(port=8000, routing_region="us-east", target_concurrency=0, serialized=True)
    class ZeroConcurrencyServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = ZeroConcurrencyServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert function_def.target_concurrent_inputs == 0


def test_server_rejects_negative_target_concurrency():
    with pytest.raises(InvalidError, match="must be a non-negative integer"):
        app = modal.App("server-negative-target-concurrency-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", target_concurrency=-1, serialized=True)
        class NegativeConcurrencyServer:
            pass


def test_server_rejects_negative_exit_grace_period():
    with pytest.raises(InvalidError, match="must be non-negative"):
        app = modal.App("server-negative-exit-grace-period-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", exit_grace_period=-1, serialized=True)
        class NegativeExitGracePeriodServer:
            pass


def test_server_rejects_too_large_exit_grace_period():
    with pytest.raises(InvalidError, match="must not exceed 3600 seconds"):
        app = modal.App("server-large-exit-grace-period-test", include_source=False)

        @app.server(port=8000, routing_region="us-east", exit_grace_period=3601, serialized=True)
        class LargeExitGracePeriodServer:
            pass


def test_server_with_volumes(client, servicer):
    """Test that servers can mount volumes."""
    app = modal.App("server-volumes-test", include_source=False)
    vol = modal.Volume.from_name("test-volume", create_if_missing=True)

    @app.server(port=8000, routing_region="us-east", volumes={"/data": vol}, serialized=True)
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

    @app.server(port=8000, routing_region="us-east", secrets=[secret], serialized=True)
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

    @app.server(port=8000, routing_region="us-east", image=custom_image, serialized=True)
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

    @app.server(port=8000, routing_region="us-east", memory=2048, cpu=4.0, serialized=True)
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


def test_server_routing_region(client, servicer):
    """Test that servers configure a single proxy region."""
    app = modal.App("server-routing-region-test", include_source=False)

    @app.server(port=8000, routing_region="us-east", serialized=True)
    class RoutingRegionServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = RoutingRegionServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert list(function_def.http_config.proxy_regions) == ["us-east"]


# =============================================================================
# Integration Tests
# =============================================================================


def test_server_creates_class_object(client, servicer):
    """Test that deploying a server creates the expected objects."""
    app = modal.App("server-objects-test", include_source=False)

    @app.server(port=8000, routing_region="us-east", serialized=True)
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

    @app.server(port=8000, routing_region="us-east", serialized=True)
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

    @app.server(port=8000, routing_region="us-east", serialized=True)
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

    @app.server(port=8000, routing_region="us-east", serialized=True)
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

    @app.server(port=8000, routing_region="us-east", serialized=True)
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
