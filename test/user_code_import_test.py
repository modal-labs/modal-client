# Copyright Modal Labs 2024
from unittest.mock import MagicMock

from modal._runtime import user_code_imports
from modal._utils.async_utils import synchronizer
from modal.image import _Image
from modal_proto import api_pb2


def test_import_function(supports_dir, monkeypatch):
    monkeypatch.syspath_prepend(supports_dir)
    fun = api_pb2.Function(module_name="user_code_import_samples.func", function_name="f")
    service = user_code_imports.import_single_function_service(
        fun,
        None,
        None,
    )
    assert len(service.service_deps) == 1
    assert type(service.service_deps[0]) is _Image
    assert service.app

    assert service.user_cls_instance is None

    # TODO (elias): shouldn't have to pass the function definition again!
    io_manager = MagicMock()  # shouldn't actually be used except by web endpoints - indicates some need for refactoring
    finalized_funcs = service.get_finalized_functions(fun, container_io_manager=io_manager)
    assert len(finalized_funcs) == 1
    finalized_func = finalized_funcs[""]
    assert finalized_func.is_async is False
    assert finalized_func.is_generator is False
    assert finalized_func.data_format == api_pb2.DATA_FORMAT_PICKLE
    assert finalized_func.lifespan_manager is None
    container_callable = finalized_func.callable
    assert container_callable("world") == "hello world"


def test_import_function_undecorated(supports_dir, monkeypatch):
    import test.supports.user_code_import_samples.func

    fun = api_pb2.Function(module_name="test.supports.user_code_import_samples.func", function_name="undecorated_f")
    service = user_code_imports.import_single_function_service(
        fun,
        None,
        None,
    )
    assert service.service_deps is None  # undecorated - can't get code deps
    # can't get app via the decorator attachment, falls back to checking global registry of apps/names
    assert service.app is synchronizer._translate_in(test.supports.user_code_import_samples.func.app)


def test_import_class(monkeypatch, supports_dir, client):
    monkeypatch.syspath_prepend(supports_dir)
    function_def = api_pb2.Function(
        module_name="user_code_import_samples.cls",
        function_name="C.*",
    )
    service = user_code_imports.import_class_service(
        function_def,
        service_function_hydration_data=api_pb2.Object(
            object_id="fu-123",
        ),
        class_id="cs-123",
        client=client,
        ser_user_cls=None,
        cls_args=(),
        cls_kwargs={},
    )
    assert len(service.service_deps) == 1
    assert type(service.service_deps[0]) is _Image

    assert service.app

    from user_code_import_samples.cls import UndecoratedC  # type: ignore

    assert isinstance(service.user_cls_instance, UndecoratedC)

    # TODO (elias): shouldn't have to pass the function definition again!
    io_manager = MagicMock()  # shouldn't actually be used except by web endpoints - indicates some need for refactoring
    finalized_funcs = service.get_finalized_functions(function_def, container_io_manager=io_manager)
    io_manager.assert_not_called()
    assert len(finalized_funcs) == 4

    for finalized in finalized_funcs.values():
        assert finalized.is_async is False
        assert finalized.is_generator is False
        assert finalized.data_format == api_pb2.DATA_FORMAT_PICKLE
        assert finalized.lifespan_manager is None

    finalized_1, finalized_2, self_ref = finalized_funcs["f"], finalized_funcs["f2"], finalized_funcs["self_ref"]
    assert finalized_1.callable("world") == "hello world"
    assert finalized_2.callable("world") == "other world"
    callable_self = self_ref.callable()
    assert isinstance(callable_self, UndecoratedC)  # Arguably this could/should be an Obj instead?


# TODO: add test cases for serialized functions, web endpoints, explicit/implicit generators etc.
#   with and without decorators in globals scope...
