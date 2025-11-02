# Copyright Modal Labs 2025
import pytest
import threading

import modal
import modal.experimental
from modal import App
from modal._utils.flash_utils import get_flash_configs
from modal.cls import Cls
from modal.exception import InvalidError

flash_app_default = App("flash-app-default")


@flash_app_default.cls()
class FlashClassDefault:
    @modal.enter()
    @modal.experimental.flash_web_server(8080, region=True, target_concurrent_requests=10, exit_grace_period=10)
    def serve(self):
        self.process = modal.experimental.flash_process(["python3", "-m", "http.server", "8080"])


def test_flash_web_server_basic_functionality(client):
    """Test basic flash_web_server decorator functionality."""
    with flash_app_default.run(client=client):
        flash_configs = get_flash_configs(FlashClassDefault)
        assert len(flash_configs) == 1
        assert flash_configs[0].port == 8080
        assert flash_configs[0].region is True
        assert flash_configs[0].target_concurrent_requests == 10
        assert flash_configs[0].exit_grace_period == 10

        # serve_method = FlashClassDefault.serve
        # from modal._partial_function import _PartialFunctionFlags

        # assert isinstance(serve_method, _PartialFunction)
        # assert serve_method.flags & _PartialFunctionFlags.FLASH_WEB_INTERFACE


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
    assert servicer.app_functions[class_function_id].experimental_options["flash"] == "True"
    assert servicer.app_functions[class_function_id].method_definitions_set == True
    assert servicer.app_functions[class_function_id].startup_timeout_secs == 300
    assert servicer.app_functions[class_function_id].app_name == "flash-app-default"
    assert servicer.app_functions[class_function_id]._experimental_concurrent_cancellations == True


flash_cls_with_enter_app = App("flash-cls-with-enter-app")


@flash_cls_with_enter_app.cls(
    enable_memory_snapshot=True,
)
class FlashClsWithEnter:
    local_thread_id: str = modal.parameter()
    post_snapshot_thread_id: str = modal.parameter()
    entered: bool = modal.parameter(default=False)
    entered_post_snapshot: bool = modal.parameter(default=False)

    @modal.enter(snap=True)
    @modal.experimental.flash_web_server(8001, region=True)
    def enter(self):
        self.entered = True
        assert threading.current_thread().name == self.local_thread_id
        self.process = modal.experimental.flash_process(["python3", "-m", "http.server", "8001"])

    @modal.enter(snap=False)
    def enter_post_snapshot(self):
        self.entered_post_snapshot = True
        assert threading.current_thread().name == self.local_thread_id

    @modal.method()
    def modal_method(self, y: int) -> int:
        return y**2


def test_enter_on_modal_flash_is_executed():
    """Test enter on modal flash is executed."""
    obj = FlashClsWithEnter(
        local_thread_id=threading.current_thread().name, post_snapshot_thread_id=threading.current_thread().name
    )
    assert obj.modal_method.local(7) == 49
    assert obj.local_thread_id == threading.current_thread().name
    assert obj.entered

flash_params_override_app = App("flash-params-override")
@flash_params_override_app.cls(experimental_options={"flash": "us-east"})
@modal.concurrent(target_inputs=10)
class FlashParamsOverrideClass:
    @modal.experimental.flash_web_server(8080, region="us-west", target_concurrent_requests=11)
    def serve(self):
        return "Flash with params override"

def test_flash_params_override_experimental_options(client, servicer):
    """Test decorator flash params override experimental options in app.cls()."""
    with flash_params_override_app.run(client=client):
        assert isinstance(FlashParamsOverrideClass, Cls)
        app_id = flash_params_override_app.app_id

        objects = servicer.app_objects[app_id]
        class_function_id = objects["FlashParamsOverrideClass.*"]

        assert servicer.app_functions[class_function_id].target_concurrent_inputs == 11
        assert servicer.app_functions[class_function_id].experimental_options["flash"] == "us-west"

# def test_partial_function_descriptors(client):
#     test whether metadata is kept correctly


def test_flash_class_with_options(client, servicer):
    """Test flash classes work with .with_options()."""
    app = App("flash-options")

    @app.cls(serialized=True)
    class FlashOptionsClass:
        @modal.experimental.flash_web_server(8080)
        def serve(self):
            return "Hello Flash"

    with app.run(client=client):
        flash_configs = get_flash_configs(FlashOptionsClass)
        assert len(flash_configs) == 1
        assert flash_configs[0].port == 8080


def test_flash_empty_class_no_configs():
    """Test that classes without flash decorators return empty configs."""
    empty_app = App("flash-empty")

    @empty_app.cls(serialized=True)
    class EmptyClass:
        @modal.method()
        def regular_method(self):
            return "No flash here"

    flash_configs = get_flash_configs(EmptyClass)
    assert len(flash_configs) == 0


def test_flash_class_inheritance():
    """Test flash decorators work with class inheritance."""
    app = App("flash-inheritance")

    class BaseFlashClass:
        @modal.experimental.flash_web_server(8080)
        def serve(self):
            return "Base flash"

    @app.cls(serialized=True)
    class DerivedFlashClass(BaseFlashClass):
        pass

    flash_configs = get_flash_configs(DerivedFlashClass)
    assert len(flash_configs) == 1
    assert flash_configs[0].port == 8080


def test_flash_no_port_parameter_error():
    """Test that flash_web_server requires a port parameter."""
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'port'"):
        flash_compatibility_app = App("flash-compatibility")

        @flash_compatibility_app.cls()
        class FlashCompatClass:
            @modal.experimental.flash_web_server()  # type: ignore  # Missing required port parameter
            def serve(self):
                return "Compatible"


def test_flash_validate_obj_compatibility(caplog):
    """Test that flash_web_server validates object compatibility."""
    with pytest.raises(
        InvalidError,
        match="Multiple flash objects are not yet supported, please only specify a single flash object.",
    ):
        flash_compatibility_app = App("flash-compatibility")

        @flash_compatibility_app.cls()
        class FlashCompatClass:
            @modal.experimental.flash_web_server(8084, region=True)
            def serve(self):
                return "Compatible"

            @modal.experimental.flash_web_server(8085, region=True)
            def serve2(self):
                return "Not Compatible"
