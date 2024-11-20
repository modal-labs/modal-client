# Copyright Modal Labs 2024
from modal.image import _Image
from modal.runtime import user_code_imports
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


def test_import_function_serialized():
    # TODO: implement
    pass


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
    assert len(service.code_deps) == 0  # undecorated - can't get code deps
    # can't reliably get app - this is deferred to a name based lookup later in the container entrypoint
    assert service.app is None


def test_import_class():
    # TODO: implement
    pass


def test_import_serialized():
    # TODO: implement
    pass


def test_import_class_undecorated():
    # TODO: implement
    pass
