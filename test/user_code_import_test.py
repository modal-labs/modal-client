# Copyright Modal Labs 2024
from unittest.mock import MagicMock

from modal._runtime import user_code_imports
from modal.image import _Image
from modal_proto import api_pb2


def test_import_function(supports_dir, monkeypatch):
    monkeypatch.syspath_prepend(supports_dir)
    fun = api_pb2.Function(module_name="user_code_import_samples.func", function_name="f")
    service = user_code_imports.import_single_function_service(
        fun,
        None,
        None,
        None,
        None,
    )
    assert len(service.code_deps) == 1
    assert type(service.code_deps[0]) is _Image
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
    monkeypatch.syspath_prepend(supports_dir)
    fun = api_pb2.Function(module_name="user_code_import_samples.func", function_name="undecorated_f")
    service = user_code_imports.import_single_function_service(
        fun,
        None,
        None,
        None,
        None,
    )
    assert service.code_deps is None  # undecorated - can't get code deps
    # can't reliably get app - this is deferred to a name based lookup later in the container entrypoint
    assert service.app is None


def test_import_class(monkeypatch, supports_dir):
    monkeypatch.syspath_prepend(supports_dir)
    fun = api_pb2.Function(
        module_name="user_code_import_samples.cls",
        function_name="C.*",
    )
    service = user_code_imports.import_class_service(
        fun,
        None,
        (),
        {},
    )
    assert len(service.code_deps) == 1
    assert type(service.code_deps[0]) is _Image

    assert service.app

    from user_code_import_samples.cls import UndecoratedC  # type: ignore

    assert isinstance(service.user_cls_instance, UndecoratedC)

    # TODO (elias): shouldn't have to pass the function definition again!
    io_manager = MagicMock()  # shouldn't actually be used except by web endpoints - indicates some need for refactoring
    finalized_funcs = service.get_finalized_functions(fun, container_io_manager=io_manager)
    assert len(finalized_funcs) == 2

    for finalized in finalized_funcs.values():
        assert finalized.is_async is False
        assert finalized.is_generator is False
        assert finalized.data_format == api_pb2.DATA_FORMAT_PICKLE
        assert finalized.lifespan_manager is None

    finalized_1, finalized_2 = finalized_funcs["f"], finalized_funcs["f2"]
    assert finalized_1.callable("world") == "hello world"
    assert finalized_2.callable("world") == "other world"


# TODO: add test cases for serialized functions, web endpoints, explicit/implicit generators etc.
#   with and without decorators in globals scope...
