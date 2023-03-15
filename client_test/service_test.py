# Copyright Modal Labs 2022
import cloudpickle

import modal
from modal_proto import api_pb2

stub = modal.Stub()


@stub.service(cpu=42)
class Foo:
    def bar(self, x):
        return x**3


def test_run_class(client, servicer):
    assert servicer.n_functions == 0
    with stub.run(client=client) as app:
        pass

    assert servicer.n_functions == 1
    (function_id,) = servicer.app_functions.keys()
    function = servicer.app_functions[function_id]
    assert function.function_name == "Foo.bar"
    objects = servicer.app_objects[app.app_id]
    assert objects == {"Foo.bar": function_id}


def test_run_class_serialized(client, servicer):
    class FooSer:
        def bar(self, x):
            return x**3

    stub_ser = modal.Stub()
    ret = stub_ser.service(cpu=42, serialized=True)(FooSer)

    assert servicer.n_functions == 0
    with stub_ser.run(client=client) as app:
        pass

    assert servicer.n_functions == 1
    (function_id,) = servicer.app_functions.keys()
    function = servicer.app_functions[function_id]
    assert function.function_name.endswith("FooSer.bar")  # because it's defined in a local scope
    assert function.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED
    cls = cloudpickle.loads(function.class_serialized)
    fun = cloudpickle.loads(function.function_serialized)

    # Create bound method
    obj = cls()
    meth = fun.__get__(obj, cls)

    # Make sure it's callable
    assert meth(100) == 1000000
