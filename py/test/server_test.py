# Copyright Modal Labs 2025
import contextlib
import pytest
import re
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from unittest import mock

import aiohttp.web

import modal
from modal._serialization import deserialize
from modal._server import _Server, _ServerSessionsManager
from modal._utils.async_utils import synchronizer
from modal._utils.http_utils import ClientSessionRegistry, run_temporary_http_server
from modal.exception import ExecutionError, InvalidError, NotFoundError
from modal.runner import deploy_app
from modal.server import Server
from modal.types import CloudBucketMountInfo, ServerInfo, ServerSessionCredentials, VolumeMountInfo
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


def test_sessioned_server_registration(client, servicer):
    app = modal.App("sessioned-server-test", include_source=False)

    @app.server(port=8000, routing_region="us-east", serialized=True)
    @modal.sessioned()
    class SessionedServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        function_id = SessionedServer._get_service_function().object_id  # type: ignore[attr-defined]
        assert servicer.app_functions[function_id].is_sessioned


def test_sessioned_decorator_only_allowed_on_servers():
    with pytest.raises(InvalidError, match=r"`@modal\.sessioned\(\)` can only be used with `@app\.server\(\)`"):
        app = modal.App("sessioned-cls-test", include_source=False)

        @app.cls()
        @modal.sessioned()
        class SessionedCls:
            pass

    with pytest.raises(InvalidError, match="Server class"):
        app = modal.App("sessioned-function-test", include_source=False)

        @app.function()
        @modal.sessioned()
        def sessioned_function():
            pass


@pytest.mark.asyncio
async def test_server_sessions():
    requests = []
    start_error = False

    async def handle_start(request):
        requests.append(request)
        if start_error:
            return aiohttp.web.Response(status=401, text="unauthorized")
        return aiohttp.web.json_response({"session_id": "se-123", "token": "sess"})

    async def handle_terminate(request):
        requests.append(request)
        return aiohttp.web.Response()

    app = aiohttp.web.Application()
    app.add_routes(
        [
            aiohttp.web.post("/_modal/sessions/start", handle_start),
            aiohttp.web.post("/_modal/sessions/terminate", handle_terminate),
        ]
    )

    class FakeFunction:
        async def _get_flash_auth_token(self):
            return "flash-jwt"

    class FakeServer:
        _is_sessioned = True

        def _is_local(self):
            return False

        def _get_service_function(self):
            return FakeFunction()

        async def get_url(self):
            return http_url

    async with aiohttp.ClientSession() as client_session:
        with mock.patch.object(ClientSessionRegistry, "get_session", return_value=client_session):
            async with run_temporary_http_server(app) as http_url:
                sessions = _ServerSessionsManager(cast(_Server, FakeServer()))
                session = await sessions.start(idle_timeout=10)
                assert session == ServerSessionCredentials(session_id="se-123", token="sess")
                assert requests[0].path == "/_modal/sessions/start"
                assert requests[0].headers["Modal-Authorization"] == "Bearer flash-jwt"
                assert requests[0].headers["x-modal-server-session-idle-timeout"] == "10"

                await sessions.terminate(session.token)
                assert requests[1].path == "/_modal/sessions/terminate"
                assert requests[1].headers["Modal-Authorization"] == "Bearer flash-jwt"
                assert requests[1].headers["x-modal-server-session-token"] == "sess"

                start_error = True
                with pytest.raises(
                    ExecutionError,
                    match=r"Failed to start session: status 401 Unauthorized$",
                ):
                    await sessions.start()


@pytest.mark.parametrize("body", ["not json", '{"session_id": "se-123"}'])
@pytest.mark.asyncio
async def test_sessions_start_malformed_response(body):
    async def handle_start(request):
        return aiohttp.web.Response(status=200, text=body)

    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.post("/_modal/sessions/start", handle_start)])

    class FakeFunction:
        async def _get_flash_auth_token(self):
            return "flash-jwt"

    class FakeServer:
        _is_sessioned = True

        def _is_local(self):
            return False

        def _get_service_function(self):
            return FakeFunction()

        async def get_url(self):
            return http_url

    async with aiohttp.ClientSession() as client_session:
        with mock.patch.object(ClientSessionRegistry, "get_session", return_value=client_session):
            async with run_temporary_http_server(app) as http_url:
                sessions = _ServerSessionsManager(cast(_Server, FakeServer()))
                with pytest.raises(ExecutionError, match="unexpected response"):
                    await sessions.start()


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
    assert servicer.app_functions[server_function_id].autoscaler_settings.target_concurrency_float == 10.0
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
    experimental_options={"flash": "us-east", "priority": "low"},
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

        assert servicer.app_functions[server_function_id].autoscaler_settings.target_concurrency_float == 11.0
        assert servicer.app_functions[server_function_id].experimental_options["flash"] == "us-east"
        assert servicer.app_functions[server_function_id].experimental_options["priority"] == "low"


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
    """Test that @modal.clustered() works with @app.server().

    Regression test: @modal.clustered() wraps the class in a _PartialFunction,
    which caused validate_wrapped_user_cls_decorators to fail on inspect.isclass().
    """
    app = modal.App("server-clustered-test", include_source=False)

    @app.server(port=8000, routing_region="us-east", serialized=True)
    @modal.clustered(size=2)
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
    assert not my_server._get_service_function()._is_hydrated
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


def test_server_from_name_hydrates_service_function_app_id(client, servicer):
    server_app.deploy(client=client)

    my_server = Server.from_name("server-test-app", "BasicServer", client=client)
    my_server.hydrate(client=client)

    service_function = my_server._get_service_function()
    service_function_impl = synchronizer._translate_in(service_function)
    assert service_function_impl._app_id == servicer.function_id_to_app_id[service_function.object_id]  # type: ignore[attr-defined]


def test_hydrated_nonsessioned_server_rejects_session_start(client, servicer):
    server_app.deploy(client=client)
    server = Server.from_name("server-test-app", "BasicServer", client=client)
    server.hydrate(client=client)

    impl = synchronizer._translate_in(server)
    assert impl._is_sessioned is False

    with mock.patch("modal._server._post_session_control") as post:
        with pytest.raises(InvalidError, match=r"requires `@modal\.sessioned\(\)`"):
            server.sessions.start()
        post.assert_not_called()


@pytest.mark.parametrize("op", ["start", "terminate"])
def test_lazy_nonsessioned_server_rejects_session_ops(client, servicer, op):
    server_app.deploy(client=client)
    server = Server.from_name("server-test-app", "BasicServer", client=client)
    assert synchronizer._translate_in(server)._is_sessioned is None

    with mock.patch("modal._server._post_session_control") as post:
        with pytest.raises(InvalidError, match=r"requires `@modal\.sessioned\(\)`"):
            if op == "start":
                server.sessions.start()
            else:
                server.sessions.terminate("tok")
        post.assert_not_called()


def test_server_from_name_failed_lookup_error(client, servicer):
    """Test that Server.from_name() raises NotFoundError with helpful message."""
    with pytest.raises(NotFoundError, match="Lookup failed.*MyServer.*my-nonexistent-app"):
        Server.from_name("my-nonexistent-app", "MyServer", client=client).hydrate()


def test_server_from_name_with_environment(client, servicer):
    """Test that Server.from_name() with environment_name includes it in error message."""
    with pytest.raises(NotFoundError, match="some-env"):
        Server.from_name("my-nonexistent-app", "MyServer", environment_name="some-env", client=client).hydrate()


# =============================================================================
# from_id Tests
# =============================================================================


def test_server_from_id(client, servicer):
    server_app.deploy(client=client)
    server_id = BasicServer.object_id  # type: ignore

    with servicer.intercept() as ctx:
        server = Server.from_id(server_id, client=client)
        service_function = server._get_service_function()
        service_function_impl = synchronizer._translate_in(service_function)
        assert not service_function_impl._is_hydrated
        assert not ctx.get_requests("FunctionGetById")

        server.hydrate()

        (request,) = ctx.get_requests("FunctionGetById")
        assert request.function_id == server_id
        assert server.object_id == server_id
        assert service_function_impl._is_hydrated


def test_server_from_id_failed_lookup(client):
    with pytest.raises(NotFoundError, match="Lookup failed for Server 'fu-does-not-exist'"):
        Server.from_id("fu-does-not-exist", client=client).hydrate()


@pytest.mark.parametrize(
    ("function", "error"),
    [
        (api_pb2.FunctionData(), "is a Function"),
        (api_pb2.FunctionData(is_class=True), "is a Cls"),
    ],
)
def test_server_from_id_rejects_other_types(client, servicer, function, error):
    async def function_get_by_id(servicer, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.FunctionGetByIdResponse(function=function))

    with servicer.intercept() as ctx:
        ctx.set_responder("FunctionGetById", function_get_by_id)
        with pytest.raises(InvalidError, match=error) as exc_info:
            Server.from_id("fu-123", client=client).hydrate()


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


def test_server_autoscaler_settings_from_proto_prefers_float():
    from modal.types import ServerAutoscalerSettings

    assert (
        ServerAutoscalerSettings._from_proto(
            api_pb2.AutoscalerSettings(target_concurrency_float=2.5)
        ).target_concurrency
        == 2.5
    )
    assert (
        ServerAutoscalerSettings._from_proto(api_pb2.AutoscalerSettings(target_concurrency=3)).target_concurrency == 3.0
    )
    assert (
        ServerAutoscalerSettings._from_proto(
            api_pb2.AutoscalerSettings(target_concurrency=3, target_concurrency_float=1.5)
        ).target_concurrency
        == 1.5
    )
    assert ServerAutoscalerSettings._from_proto(api_pb2.AutoscalerSettings()).target_concurrency is None


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
        settings = update_autoscaler(min_containers=1, max_containers=5, target_concurrency=20)  # type: ignore[call-arg]
        assert settings.target_concurrency == 20
        assert servicer.app_functions[function_id].autoscaler_settings.target_concurrency_float == 20
        # Overrides are tracked on autoscaler_settings; the static definition field is unchanged.
        assert servicer.app_functions[function_id].target_concurrent_inputs == 0
        assert not servicer.app_functions[function_id].autoscaler_settings.HasField("target_concurrency")

        settings = update_autoscaler(target_concurrency=0)  # type: ignore[call-arg]

        settings = update_autoscaler(target_concurrency=0.5)  # type: ignore[call-arg]
        assert settings.target_concurrency == 1.0
        assert servicer.app_functions[function_id].autoscaler_settings.target_concurrency_float == 1.0

        settings = update_autoscaler(target_concurrency=1.234)  # type: ignore[call-arg]
        assert settings.target_concurrency == 1.23
        assert servicer.app_functions[function_id].autoscaler_settings.target_concurrency_float == 1.23

        settings = update_autoscaler(target_concurrency=0)  # type: ignore[call-arg]

    f = servicer.app_functions[function_id]

    assert settings.min_containers == f.autoscaler_settings.min_containers == 1
    assert settings.max_containers == f.autoscaler_settings.max_containers == 5
    assert f.autoscaler_settings.HasField("target_concurrency_float")
    assert settings.target_concurrency == f.autoscaler_settings.target_concurrency_float == 0
    assert f.target_concurrent_inputs == 0
    assert not f.autoscaler_settings.HasField("target_concurrency")


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

        assert function_def.autoscaler_settings.target_concurrency_float == 50.0


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

        assert function_def.autoscaler_settings.target_concurrency_float == 0.0


def test_server_max_concurrency(client, servicer):
    app = modal.App("server-max-concurrency-test", include_source=False)

    @app.server(
        port=8000,
        routing_region="us-east",
        target_concurrency=50.5,
        max_concurrency=100,
        serialized=True,
    )
    class MaxConcurrencyServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        function_id = MaxConcurrencyServer._get_service_function().object_id  # type: ignore[attr-defined]
        function_def = servicer.app_functions[function_id]
        assert function_def.autoscaler_settings.target_concurrency_float == 50.5
        assert function_def.max_concurrent_inputs == 100


@pytest.mark.parametrize(
    ("max_concurrency", "match"),
    [
        (True, "must be a number"),
        (1.5, "must be an integer"),
        (-1, "must be non-negative"),
        ("100", "must be a number"),
    ],
)
def test_server_rejects_invalid_max_concurrency(max_concurrency, match):
    with pytest.raises(InvalidError, match=match):
        app = modal.App("server-invalid-max-concurrency-test", include_source=False)

        @app.server(
            port=8000,
            routing_region="us-east",
            max_concurrency=cast(Any, max_concurrency),
            serialized=True,
        )
        class InvalidMaxConcurrencyServer:
            pass


def test_server_allows_zero_max_concurrency(client, servicer):
    app = modal.App("server-zero-max-concurrency-test", include_source=False)

    @app.server(
        port=8000,
        routing_region="us-east",
        target_concurrency=10,
        max_concurrency=0,
        serialized=True,
    )
    class UnlimitedConcurrencyServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        function_id = UnlimitedConcurrencyServer._get_service_function().object_id  # type: ignore[attr-defined]
        function_def = servicer.app_functions[function_id]
        assert function_def.autoscaler_settings.target_concurrency_float == 10
        assert function_def.max_concurrent_inputs == 0


def test_server_rejects_target_concurrency_above_max_concurrency():
    with pytest.raises(InvalidError, match="cannot be greater than `max_concurrency`"):
        app = modal.App("server-target-above-max-concurrency-test", include_source=False)

        @app.server(
            port=8000,
            routing_region="us-east",
            target_concurrency=10.5,
            max_concurrency=10,
            serialized=True,
        )
        class TargetAboveMaxConcurrencyServer:
            pass


def test_server_rejects_duplicate_max_concurrency_configuration():
    with pytest.raises(InvalidError, match="cannot be set both"):
        app = modal.App("server-duplicate-max-concurrency-test", include_source=False)

        @app.server(
            port=8000,
            routing_region="us-east",
            max_concurrency=10,
            experimental_options={"max_concurrency": 3},
            serialized=True,
        )
        class DuplicateMaxConcurrencyServer:
            pass


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (0.5, 1.0),
        (1.234, 1.23),
        (2.5, 2.5),
    ],
)
def test_server_target_concurrency_clamped_and_rounded(client, servicer, requested, expected):
    """Values in (0, 1) clamp to 1; other values round to the nearest hundredth."""
    app = modal.App(f"server-target-concurrency-norm-{requested}", include_source=False)

    @app.server(port=8000, routing_region="us-east", target_concurrency=requested, serialized=True)
    class NormConcurrencyServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        function_id = NormConcurrencyServer._get_service_function().object_id  # type: ignore[attr-defined]
        assert servicer.app_functions[function_id].autoscaler_settings.target_concurrency_float == expected


def test_server_rejects_negative_target_concurrency():
    with pytest.raises(InvalidError, match="must be non-negative"):
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


def test_server_compute_region(client, servicer):
    """Test that `compute_region` configures the scheduler placement regions."""
    app = modal.App("server-compute-region-test", include_source=False)

    @app.server(port=8000, routing_region="us-east", compute_region=["us-east-1", "us-west-2"], serialized=True)
    class ComputeRegionServer:
        @modal.enter()
        def start(self):
            pass

    with app.run(client=client):
        service_function = ComputeRegionServer._get_service_function()  # type: ignore[attr-defined]
        function_id = service_function.object_id
        function_def = servicer.app_functions[function_id]

        assert list(function_def.scheduler_placement.regions) == ["us-east-1", "us-west-2"]


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


server_info_app = modal.App()


@server_info_app.server(port=8000, routing_region="us-east", serialized=True)
class InfoServer:
    @modal.enter()
    def start(self):
        pass


def test_server_info_local():
    info: modal.types.ServerInfo = InfoServer.info()  # type: ignore[attr-defined]
    assert not info.sessioned
    assert info.http_info.proxy_regions == ["us-east"]
    assert info.http_info.port == 8000
    assert info.http_info.unauthenticated == False
    assert info.http_info.h2_enabled == False
    assert info.timeout == 300
    assert info.max_retries is None

    assert not InfoServer._get_service_function()._is_hydrated  # type: ignore[attr-defined]


def test_server_info_remote(client, servicer):
    server = Server.from_name("dummy-app", "func", client=client)
    function_id = "fu-1"

    with servicer.intercept() as ctx:
        ctx.add_response(
            "FunctionGet",
            api_pb2.FunctionGetResponse(
                function_id=function_id,
                function=api_pb2.FunctionData(
                    ranked_functions=[
                        api_pb2.FunctionData.RankedFunction(
                            rank=1,
                            function=api_pb2.Function(
                                function_name="func",
                                volume_mounts=[
                                    api_pb2.VolumeMount(volume_id="vo-123", mount_path="/tmp"),
                                    api_pb2.VolumeMount(volume_id="vo-456", mount_path="/mnt", read_only=True),
                                ],
                                cloud_bucket_mounts=[
                                    api_pb2.CloudBucketMount(
                                        bucket_name="bucket-name",
                                        mount_path="/dev",
                                        bucket_type=api_pb2.CloudBucketMount.BucketType.S3,
                                    )
                                ],
                            ),
                        )
                    ],
                    is_server=True,
                    is_sessioned=True,
                    http_config=api_pb2.HTTPConfig(
                        port=1, proxy_regions=["us-west-2"], unauthenticated=True, h2_enabled=True
                    ),
                ),
            ),
        )

        info = server.info()

        assert info.sessioned

        assert info.volumes == {
            "/tmp": VolumeMountInfo(
                name=None,
                volume_id="vo-123",
                read_only=False,
                sub_path=None,
            ),
            "/mnt": VolumeMountInfo(
                name=None,
                volume_id="vo-456",
                read_only=True,
                sub_path=None,
            ),
        }

        assert info.cloud_bucket_mounts == {
            "/dev": CloudBucketMountInfo(
                bucket_name="bucket-name",
                bucket_type="s3",
                read_only=False,
                key_prefix=None,
            )
        }

        assert info.http_info.h2_enabled
        assert info.http_info.unauthenticated
        assert info.http_info.proxy_regions == ["us-west-2"]
        assert info.http_info.port == 1


def test_server_info_refresh(client):
    app = modal.App()

    @app.server(routing_region="us-east", serialized=True)
    class InfoServer:
        @modal.enter()
        def start(self):
            pass

    deploy_app(app, "test_function_info_redeploy", client=client)

    handle = InfoServer

    info: ServerInfo = handle.info()  # type: ignore[attr-defined]
    assert info.http_info.proxy_regions == ["us-east"]

    _ = app.server(routing_region="us-west", serialized=True)(InfoServer._get_user_cls())  # type: ignore[attr-defined]

    deploy_app(app, "test_function_info_redeploy", client=client)

    new_info = handle.info(refresh=True)  # type: ignore[attr-defined]
    assert new_info.http_info.proxy_regions == ["us-west"]


def test_image_info(client):
    builder_app = modal.App()
    with builder_app.run(client=client):
        modal.Image.debian_slim("3.12").build(builder_app).publish("named-image", client=client)

    app = modal.App(image=modal.Image.debian_slim("3.10"))
    anon_image = modal.Image.debian_slim("3.11").pip_install("aiohttp")
    named_image = modal.Image.from_name("named-image")

    @app.server(serialized=True)
    class ServerDefaultImage:
        @modal.enter()
        def start(self):
            pass

    @app.server(serialized=True, image=anon_image)
    class ServerAnonImage:
        @modal.enter()
        def start(self):
            pass

    @app.server(serialized=True, image=named_image)
    class ServerNamedImage:
        @modal.enter()
        def start(self):
            pass

    assert ServerDefaultImage.info().image_info == modal.types.ServerInfo.ImageInfo(None, None)  # type: ignore[attr-defined]

    # Ideally this would not have two `None`s, see todo in the image_info constructor in
    # `_functions.py`
    assert ServerAnonImage.info().image_info == modal.types.ServerInfo.ImageInfo(None, None)  # type: ignore[attr-defined]

    assert ServerNamedImage.info().image_info == modal.types.ServerInfo.ImageInfo("named-image", None)  # type: ignore[attr-defined]

    with app.run(client=client):
        assert ServerDefaultImage.info(refresh=True).image_info.image_id is not None  # type: ignore[attr-defined]
        assert ServerAnonImage.info(refresh=True).image_info.image_id is not None  # type: ignore[attr-defined]
        assert ServerNamedImage.info(refresh=True).image_info.image_id is not None  # type: ignore[attr-defined]


def _stats_distribution(unit: str, p50: float, p90: float, p99: float) -> api_pb2.StatsPercentileDistribution:
    return api_pb2.StatsPercentileDistribution(
        unit=unit,
        percentiles=[
            api_pb2.StatsPercentile(percentile_basis_points=5000, value=p50),
            api_pb2.StatsPercentile(percentile_basis_points=9000, value=p90),
            api_pb2.StatsPercentile(percentile_basis_points=9900, value=p99),
        ],
    )


def test_server_stats(client, servicer):
    server_app.deploy(client=client)
    server = Server.from_name("server-test-app", "BasicServer", client=client)
    since = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
    until = since + timedelta(hours=2)
    response = api_pb2.ServerGetTimeRangeStatsResponse(
        request_count=1284,
        request_count_by_status_code=[
            api_pb2.ServerGetTimeRangeStatsResponse.ServerStatusCodeCount(status_code=200, count=1241),
            api_pb2.ServerGetTimeRangeStatsResponse.ServerStatusCodeCount(status_code=400, count=37),
            api_pb2.ServerGetTimeRangeStatsResponse.ServerStatusCodeCount(status_code=500, count=6),
        ],
        request_rate_per_second=0.36,
        request_percentile_stats={
            "request_latency": _stats_distribution("seconds", 1.18, 3.51, 5.72),
        },
        container_percentile_stats={
            "startup_time": _stats_distribution("seconds", 5.484, 7.13, 8.25),
            "cpu_usage": _stats_distribution("cores", 0.35, 0.72, 0.9),
        },
        inference=api_pb2.ServerGetTimeRangeStatsResponse.ServerInferenceStats(
            engine=api_pb2.LLM_ENGINE_SGLANG,
            status=api_pb2.SERVER_INFERENCE_STATS_STATUS_AVAILABLE,
            percentile_stats={
                "time_to_first_token": _stats_distribution("seconds", 0.121, 0.317, 0.5),
            },
            scalar_stats={"output_tokens_per_second": 585.0},
        ),
        container_started_count=10,
        container_error_count=1,
        container_creating_at_end_count=2,
    )
    response.since.FromDatetime(since)
    response.until.FromDatetime(until)

    with servicer.intercept() as ctx:
        ctx.add_response("ServerGetTimeRangeStats", response)
        stats = server.stats(since=since, until=until, container="ta-123")

    request = ctx.pop_request("ServerGetTimeRangeStats")
    assert request.function_id == server.object_id
    assert request.since.ToDatetime(tzinfo=timezone.utc) == since
    assert request.until.ToDatetime(tzinfo=timezone.utc) == until
    assert request.container_id == "ta-123"

    assert stats.since == since
    assert stats.until == until
    assert stats.request_count == 1284
    assert stats.request_count_by_status_code == {200: 1241, 400: 37, 500: 6}
    assert stats.request_rate_per_second == 0.36
    assert stats.request_percentile_stats["request_latency"].unit == "seconds"
    assert stats.container_percentile_stats["startup_time"].unit == "seconds"
    assert stats.container_percentile_stats["cpu_usage"].unit == "cores"
    assert stats.container_started_count == 10
    assert stats.container_error_count == 1
    assert stats.container_creating_at_end_count == 2
    assert stats.inference is not None
    assert stats.inference.engine == "sglang"
    assert stats.inference.status == "available"
    assert stats.inference.percentile_stats["time_to_first_token"].unit == "seconds"
    assert stats.inference.scalar_stats == {"output_tokens_per_second": 585.0}


def test_server_stats_default_time_range(client, servicer):
    server_app.deploy(client=client)
    server = Server.from_name("server-test-app", "BasicServer", client=client)
    response_since = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
    response_until = response_since + timedelta(hours=1)
    response = api_pb2.ServerGetTimeRangeStatsResponse()
    response.since.FromDatetime(response_since)
    response.until.FromDatetime(response_until)

    before = datetime.now(timezone.utc)
    with servicer.intercept() as ctx:
        ctx.add_response("ServerGetTimeRangeStats", response)
        server.stats()
    after = datetime.now(timezone.utc)

    request = ctx.pop_request("ServerGetTimeRangeStats")
    requested_since = request.since.ToDatetime(tzinfo=timezone.utc)
    requested_until = request.until.ToDatetime(tzinfo=timezone.utc)
    assert before <= requested_until <= after
    assert requested_until - requested_since == timedelta(hours=1)


def test_server_stats_rejects_invalid_time_range(client, servicer):
    server_app.deploy(client=client)
    server = Server.from_name("server-test-app", "BasicServer", client=client)
    now = datetime.now(timezone.utc)

    with servicer.intercept() as ctx:
        with pytest.raises(InvalidError, match="must be before"):
            server.stats(since=now, until=now)

    assert ctx.get_requests("ServerGetTimeRangeStats") == []
