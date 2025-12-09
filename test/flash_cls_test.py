# Copyright Modal Labs 2025
import pytest
import subprocess

import modal
import modal.experimental
from modal import App
from modal.cls import Cls
from modal.exception import InvalidError

flash_app_default = App("flash-app-default")


@flash_app_default.cls()
@modal.concurrent(target_inputs=10)
@modal.experimental.http_server(8080, proxy_regions=["us-east", "us-west"], exit_grace_period=10)
class FlashClassDefault:
    @modal.enter()
    def serve(self):
        self.process = subprocess.Popen(["python3", "-m", "http.server", "8080"])


app = App("flash-app-2")


@app.cls(
    min_containers=1,
    image=modal.Image.debian_slim().pip_install("fastapi", "uvicorn"),
)
@modal.concurrent(target_inputs=100)
@modal.experimental.http_server(
    8080,
    proxy_regions=["us-east", "us-west", "ap-south"],
    startup_timeout=10,
    exit_grace_period=10,
    concurrent_requests=20,
)
class FlashClass:
    @modal.enter()
    def start(self):
        self.process = subprocess.Popen(["python3", "-m", "http.server", "8080"])


def test_http_server_basic_functionality(client, servicer):
    """Test basic http_server decorator functionality."""
    with flash_app_default.run(client=client):
        service_function = FlashClassDefault._get_class_service_function()  # type: ignore
        function_id = service_function.object_id

        function_def = servicer.app_functions[function_id]
        http_config = function_def.http_config

        assert http_config is not None
        assert http_config.port == 8080
        assert list(http_config.proxy_regions) == ["us-east", "us-west"]
        assert http_config.exit_grace_period == 10
        assert http_config.concurrent_requests == 0  # not configured

    with app.run(client=client):
        service_function = FlashClass._get_class_service_function()  # type: ignore
        function_id = service_function.object_id

        function_def = servicer.app_functions[function_id]
        http_config = function_def.http_config

        assert http_config.concurrent_requests == 20


def test_run_class(client, servicer):
    """Test running flash class params are set correctly."""
    assert len(servicer.precreated_functions) == 0
    assert servicer.n_functions == 0
    with flash_app_default.run(client=client):
        method_handle_object_id = FlashClassDefault._get_class_service_function().object_id  # type: ignore
        assert isinstance(FlashClassDefault, Cls)
        class_id = FlashClassDefault.object_id
        app_id = flash_app_default.app_id

    assert len(servicer.classes) == 1 and set(servicer.classes) == {class_id}
    assert servicer.n_functions == 1
    objects = servicer.app_objects[app_id]
    class_function_id = objects["FlashClassDefault.*"]
    assert servicer.precreated_functions == {class_function_id}
    assert method_handle_object_id == class_function_id  # method handle object id will probably go away
    assert len(objects) == 2  # one class + one class service function
    assert objects["FlashClassDefault"] == class_id
    assert class_function_id.startswith("fu-")
    assert servicer.app_functions[class_function_id].is_class

    assert servicer.app_functions[class_function_id].module_name == "test.flash_cls_test"
    assert servicer.app_functions[class_function_id].function_name == "FlashClassDefault.*"
    assert servicer.app_functions[class_function_id].target_concurrent_inputs == 10
    assert servicer.app_functions[class_function_id].experimental_options["flash"] == ""  # empty
    assert servicer.app_functions[class_function_id].method_definitions_set == True
    assert servicer.app_functions[class_function_id].startup_timeout_secs == 300
    assert servicer.app_functions[class_function_id].app_name == "flash-app-default"
    assert servicer.app_functions[class_function_id]._experimental_concurrent_cancellations == True


def test_invalid_flash_class_decorator_on_method():
    """Test invalid flash class decorator on method."""
    with pytest.raises(
        InvalidError,
        match="The `@modal.http_server` decorator cannot be used on methods; decorate the class instead.",
    ):

        @app.cls(
            enable_memory_snapshot=True,
            serialized=True,
        )
        class InvalidFlashClassDecoratorOnMethod:
            @modal.experimental.http_server(8001, proxy_regions=["us-east", "us-west", "ap-south"])
            @modal.enter(snap=True)
            def enter(self):
                self.entered = True
                self.process = subprocess.Popen(["python3", "-m", "http.server", "8001"])


def test_invalid_flash_class_method():
    """Test invalid flash class method."""
    with pytest.raises(InvalidError, match="Callable decorators cannot be combined with web interface decorators."):

        @app.cls(
            enable_memory_snapshot=True,
            serialized=True,
        )
        @modal.experimental.http_server(8001, proxy_regions=["us-east", "us-west", "ap-south"])
        class InvalidFlashClassMethod:
            @modal.enter(snap=True)
            def enter(self):
                self.entered = True
                self.process = subprocess.Popen(["python3", "-m", "http.server", "8001"])

            @modal.method()
            def modal_method(self, y: int) -> int:
                return y**2


flash_params_override_app = App("flash-params-override")


@flash_params_override_app.cls(experimental_options={"flash": "us-east"})
@modal.experimental.http_server(8080, proxy_regions=["us-west"])
@modal.concurrent(target_inputs=11)
class FlashParamsOverrideClass:
    def serve(self):
        return "Flash with params override"


def test_flash_params_override_experimental_options(client, servicer):
    """Test experimental options and http_server decorator work together."""
    with flash_params_override_app.run(client=client):
        assert isinstance(FlashParamsOverrideClass, Cls)
        app_id = flash_params_override_app.app_id

        objects = servicer.app_objects[app_id]
        class_function_id = objects["FlashParamsOverrideClass.*"]

        assert servicer.app_functions[class_function_id].target_concurrent_inputs == 11
        assert servicer.app_functions[class_function_id].experimental_options["flash"] == "us-east"


def test_flash_class_with_options(client, servicer):
    """Test flash classes work with .with_options()."""
    app = App("flash-options")

    @app.cls(serialized=True)
    @modal.experimental.http_server(8080)
    class FlashOptionsClass:
        def serve(self):
            return "Hello Flash"

    with app.run(client=client):
        service_function = FlashOptionsClass._get_class_service_function()  # type: ignore
        function_id = service_function.object_id
        service_function_defn = servicer.app_functions[function_id]
        assert service_function_defn.http_config is not None
        assert service_function_defn.http_config.port == 8080


def test_flash_empty_class_no_configs(client, servicer):
    """Test that classes without flash decorators return empty configs."""
    empty_app = App("flash-empty")

    @empty_app.cls(serialized=True)
    class EmptyClass:
        @modal.method()
        def regular_method(self):
            return "No flash here"

    with empty_app.run(client=client):
        service_function = EmptyClass._get_class_service_function()  # type: ignore
        function_id = service_function.object_id
        service_function_defn = servicer.app_functions[function_id]
        assert not service_function_defn.HasField("http_config")


def test_flash_no_port_parameter_error():
    """Test that http_server requires a port parameter."""
    with pytest.raises(
        modal.exception.InvalidError,
        match=(
            r"Positional arguments are not allowed\. "
            r"Did you forget parentheses\? Suggestion: `@modal\.http_server\(\)`\."
        ),
    ):
        flash_compatibility_app = App("flash-compatibility")

        @flash_compatibility_app.cls()
        @modal.experimental.http_server()  # type: ignore  # Missing required port parameter
        class FlashCompatClass:
            def serve(self):
                return "Compatible"
