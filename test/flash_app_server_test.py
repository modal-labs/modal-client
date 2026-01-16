# Copyright Modal Labs 2025
import subprocess

import modal
from modal import App
from modal.server import Server

flash_app_default = App("flash-app-default")


@flash_app_default.server(
    port=8080,
    proxy_regions=["us-east", "us-west"],
    exit_grace_period=10,
    target_concurrency=10,
)
class FlashClassDefault:
    @modal.enter()
    def serve(self):
        self.process = subprocess.Popen(["python3", "-m", "http.server", "8080"])


app = App("flash-app-2")


@app.server(
    port=8080,
    proxy_regions=["us-east", "us-west", "ap-south"],
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


def test_http_server_basic_functionality(client, servicer):
    """Test basic server decorator functionality."""
    with flash_app_default.run(client=client):
        service_function = FlashClassDefault._get_service_function()  # type: ignore
        function_id = service_function.object_id

        function_def = servicer.app_functions[function_id]
        http_config = function_def.http_config

        assert http_config is not None
        assert http_config.port == 8080
        assert list(http_config.proxy_regions) == ["us-east", "us-west"]
        assert http_config.exit_grace_period == 10


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

    assert servicer.app_functions[server_function_id].module_name == "test.flash_app_server_test"
    assert servicer.app_functions[server_function_id].function_name == "FlashClassDefault"
    assert servicer.app_functions[server_function_id].target_concurrent_inputs == 10
    assert servicer.app_functions[server_function_id].method_definitions_set
    assert servicer.app_functions[server_function_id].startup_timeout_secs == 30
    assert servicer.app_functions[server_function_id].app_name == "flash-app-default"
    assert servicer.app_functions[server_function_id]._experimental_concurrent_cancellations


flash_params_override_app = App("flash-params-override")


@flash_params_override_app.server(
    port=8080,
    proxy_regions=["us-west"],
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


def test_server_http_config_is_set(client, servicer):
    """Test servers have http_config set correctly."""
    test_app = App("server-options")

    @test_app.server(port=8080, proxy_regions=["us-east"], serialized=True)
    class ServerOptionsClass:
        @modal.enter()
        def serve(self):
            pass

    with test_app.run(client=client):
        service_function = ServerOptionsClass._get_service_function()  # type: ignore
        function_id = service_function.object_id
        service_function_defn = servicer.app_functions[function_id]
        assert service_function_defn.http_config is not None
        assert service_function_defn.http_config.port == 8080
