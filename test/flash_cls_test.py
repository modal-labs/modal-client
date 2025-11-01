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
        self.process = modal.experimental.flash_process(['python3', '-m', 'http.server', '8080'])


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
@flash_cls_with_enter_app.cls(enable_memory_snapshot=True, )
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
        self.process = modal.experimental.flash_process(['python3', '-m', 'http.server', '8001'])

    @modal.enter(snap=False)
    def enter_post_snapshot(self):
        self.entered_post_snapshot = True
        assert threading.current_thread().name == self.local_thread_id

    @modal.method()
    def modal_method(self, y: int) -> int:
        return y**2

def test_enter_on_modal_flash_is_executed():
    """Test enter on modal flash is executed."""
    obj = FlashClsWithEnter(local_thread_id=threading.current_thread().name, post_snapshot_thread_id=threading.current_thread().name)
    assert obj.modal_method.local(7) == 49
    assert obj.local_thread_id == threading.current_thread().name
    assert obj.entered

# @app_method_args.cls(min_containers=5)
# class XYZ:
#     @method()
#     def foo(self): ...

#     @method()
#     def bar(self): ...


# def test_method_args(servicer, client):
#     with app_method_args.run(client=client):
#         funcs = servicer.app_functions.values()
#         assert {f.function_name for f in funcs} == {"XYZ.*"}
#         warm_pools = {f.function_name: f.autoscaler_settings.min_containers for f in funcs}
#         assert warm_pools == {"XYZ.*": 5}


# def test_cls_update_autoscaler(client, servicer):
#     app = App()

#     @app.cls(serialized=True)
#     class ClsWithMethod:
#         arg: str = modal.parameter(default="")

#         @method()
#         def bar(self): ...

#     with app.run(client=client):
#         assert len(servicer.app_functions) == 1  # only class service function
#         cls_service_fun = servicer.function_by_name("ClsWithMethod.*")
#         assert cls_service_fun.is_class
#         assert cls_service_fun.warm_pool_size == 0

#         empty_args_obj = typing.cast(modal.cls.Obj, ClsWithMethod())
#         empty_args_obj.update_autoscaler(min_containers=2, buffer_containers=1)
#         service_function_id = empty_args_obj._cached_service_function().object_id
#         service_function_defn = servicer.app_functions[service_function_id]
#         autoscaler_settings = service_function_defn.autoscaler_settings
#         assert service_function_defn.warm_pool_size == autoscaler_settings.min_containers == 2
#         assert service_function_defn._experimental_buffer_containers == autoscaler_settings.buffer_containers == 1

#         param_obj = ClsWithMethod(arg="other-instance")
#         param_obj.update_autoscaler(min_containers=5, max_containers=10)  # type: ignore
#         assert len(servicer.app_functions) == 3  # base + 2 x instance service function
#         assert cls_service_fun.warm_pool_size == 0  # base still has no warm

#         instance_service_function_id = param_obj._cached_service_function().object_id  # type: ignore
#         instance_service_defn = servicer.app_functions[instance_service_function_id]
#         instance_autoscaler_settings = instance_service_defn.autoscaler_settings
#         assert instance_service_defn.warm_pool_size == instance_autoscaler_settings.min_containers == 5
#         assert instance_service_defn.concurrency_limit == instance_autoscaler_settings.max_containers == 10

# def test_flash_params_override_experimental_options(client, servicer):
#     """Test flash params override experimental options in app.cls()."""
#     flash_params_override_app = App("flash-params-override")
#     @flash_params_override_app.cls(experimental_options={"flash": "us-east"}, serialized=True)
#     class FlashParamsOverrideClass:
#         @modal.experimental.flash_web_server(8080, region="us-west")
#         def serve(self):
#             return "Flash with params override"

#     # assert flash_params_override_app.experimental_options["flash"] == "us-west"

# def test_cls_lookup_update_autoscaler(client, servicer):
#     app = App(name := "my-cls-app")

#     @app.cls(serialized=True)
#     class ClsWithMethod:
#         arg: str = modal.parameter(default="")

#         @method()
#         def bar(self): ...

#     C_pre_deploy = ClsWithMethod()
#     with pytest.raises(ExecutionError, match="has not been hydrated"):
#         C_pre_deploy.update_autoscaler(min_containers=1)  # type: ignore

#     deploy_app(app, name, client=client)

#     C = Cls.from_name(name, "ClsWithMethod")
#     obj = C()
#     obj.update_autoscaler(min_containers=3)

#     service_function_id = obj._cached_service_function().object_id
#     assert servicer.app_functions[service_function_id].warm_pool_size == 3

#     with servicer.intercept() as ctx:
#         obj.update_autoscaler(min_containers=4)
#         assert len(ctx.get_requests("FunctionBindParams")) == 0  # We did not re-bind



# def test_partial_function_descriptors(client):
#     class Foo:
#         @modal.enter()
#         def enter_method(self):
#             pass

#         @modal.method()
#         def bar(self):
#             return "a"

#         @modal.fastapi_endpoint()
#         def web(self):
#             pass

#     assert isinstance(Foo.bar, PartialFunction)

#     assert Foo().bar() == "a"  # type: ignore   # edge case - using a non-decorated class should just return the bound original method
#     assert inspect.ismethod(Foo().bar)
#     app = modal.App()

#     modal_foo_class = app.cls(serialized=True)(Foo)

#     wrapped_method = modal_foo_class().bar
#     assert isinstance(wrapped_method, Function)

#     serialized_class = serialize(Foo)
#     revived_class = deserialize(serialized_class, client)

#     assert (
#         revived_class().bar() == "a"
#     )  # this instantiates the underlying "user_cls", so it should work basically like a normal Python class
#     assert isinstance(
#         revived_class.bar, PartialFunction
#     )  # but it should be a PartialFunction, so it keeps associated metadata!

#     # ensure that webhook metadata is kept
#     web_partial_function: _PartialFunction = synchronizer._translate_in(revived_class.web)  # type: ignore
#     assert web_partial_function.params.webhook_config
#     assert web_partial_function.params.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION



# def test_concurrent_decorator_stacked_with_method_decorator():
#     app = modal.App()

#     with pytest.raises(modal.exception.InvalidError, match="decorate the class"):

#         @app.cls(serialized=True)
#         class UsesMethodAndConcurrentDecorators:
#             @modal.method()
#             @modal.concurrent(max_inputs=10)
#             def method(self):
#                 pass


# def test_cls_get_flash_url(servicer):
#     """Test get_flash_url method on Cls.from_name instances"""
#     cls = Cls.from_name("dummy-app", "MyClass")

#     with servicer.intercept() as ctx:
#         ctx.add_response(
#             "ClassGet",
#             api_pb2.ClassGetResponse(class_id="cs-1"),
#         )
#         ctx.add_response(
#             "FunctionGet",
#             api_pb2.FunctionGetResponse(
#                 function_id="fu-1",
#                 handle_metadata=api_pb2.FunctionHandleMetadata(
#                     function_name="MyClass.*",
#                     is_method=False,
#                     _experimental_flash_urls=[
#                         "https://flash.example.com/service1",
#                         "https://flash.example.com/service2",
#                     ],
#                 ),
#             ),
#         )
#         flash_urls = cls._experimental_get_flash_urls()
#         assert flash_urls == ["https://flash.example.com/service1", "https://flash.example.com/service2"]


# # def test_flash_web_server_with_no_region_error(client, servicer):
# #     """Test flash_web_server decorator with region specification."""
# #     with pytest.raises(TypeError):

# #         @flash_app_default.cls()
# #         class FlashClassWithRegion:
# #             @flash_web_server(8080)
# #             def serve(self):
# #                 return "Hello Flash"


# # def test_flash_multiple_objects_error():
# #     """Test that multiple flash objects in one class raise an error."""

# #     with pytest.raises(InvalidError, match="Multiple flash objects are not yet supported"):

# #         @flash_app_default.cls()
# #         class MultiFlashClass:
# #             @flash_web_server(8080)
# #             def serve1(self):
# #                 pass

# #             @flash_web_server(8081)
# #             def serve2(self):
# #                 pass


# # def test_flash_with_method_decorator(client, servicer):
# #     """Test flash_web_server stacking with @method decorator."""
# #     app = App("flash-method")

# #     @app.cls()
# #     class FlashMethodClass:
# #         @modal.method()
# #         @flash_web_server(8080)
# #         def serve(self):
# #             return "Hello Flash"

# #         @modal.method()
# #         def regular_method(self):
# #             return "Regular method"

# #     with app.run(client=client):
# #         obj = FlashMethodClass()
# #         flash_configs = get_flash_configs(FlashMethodClass)
# #         assert len(flash_configs) == 1
# #         assert flash_configs[0].port == 8080


# # def test_flash_class_serialization(client, servicer):
# #     """Test flash classes work with serialization."""
# #     app = App("flash-serialized")

# #     @app.cls(serialized=True)
# #     class SerializedFlashClass:
# #         @flash_web_server(8080)
# #         def serve(self):
# #             return "Hello Serialized Flash"

# #     with app.run(client=client):
# #         obj = SerializedFlashClass()
# #         flash_configs = get_flash_configs(SerializedFlashClass)
# #         assert len(flash_configs) == 1
# #         assert flash_configs[0].port == 8080


# # def test_flash_class_with_options(client, servicer):
# #     """Test flash classes work with .with_options()."""
# #     app = App("flash-options")

# #     @app.cls()
# #     class FlashOptionsClass:
# #         @flash_web_server(8080)
# #         def serve(self):
# #             return "Hello Flash"

# #     with app.run(client=client):
# #         # Test with_options doesn't break flash functionality
# #         flash_class_with_opts = FlashOptionsClass.with_options(cpu=2, memory=1024)  # type: ignore
# #         obj = flash_class_with_opts()

# #         # Flash configs should still be accessible from the original class
# #         flash_configs = get_flash_configs(FlashOptionsClass)
# #         assert len(flash_configs) == 1
# #         assert flash_configs[0].port == 8080


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
            @modal.experimental.flash_web_server() # type: ignore  # Missing required port parameter
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

