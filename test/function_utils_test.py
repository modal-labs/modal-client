# Copyright Modal Labs 2023
import time

from grpclib import Status

from modal import method, web_endpoint
from modal._serialization import serialize_data_format
from modal._utils import async_utils
from modal._utils.function_utils import (
    FunctionInfo,
    _stream_function_call_data,
    callable_has_non_self_non_default_params,
    callable_has_non_self_params,
)
from modal_proto import api_pb2


def hasarg(a):
    ...


def noarg():
    ...


def defaultarg(a="hello"):
    ...


def wildcard_args(*wildcard_list, **wildcard_dict):
    ...


def test_is_nullary():
    assert not FunctionInfo(hasarg).is_nullary()
    assert FunctionInfo(noarg).is_nullary()
    assert FunctionInfo(defaultarg).is_nullary()
    assert FunctionInfo(wildcard_args).is_nullary()


class Cls:
    def f1(self):
        pass

    def f2(self, x):
        pass

    def f3(self, *args):
        pass

    def f4(self, x=1):
        pass


def f5():
    pass


def f6(x):
    pass


def f7(x=1):
    pass


def test_callable_has_non_self_params():
    assert not callable_has_non_self_params(Cls.f1)
    assert not callable_has_non_self_params(Cls().f1)
    assert callable_has_non_self_params(Cls.f2)
    assert callable_has_non_self_params(Cls().f2)
    assert callable_has_non_self_params(Cls.f3)
    assert callable_has_non_self_params(Cls().f3)
    assert callable_has_non_self_params(Cls.f4)
    assert callable_has_non_self_params(Cls().f4)
    assert not callable_has_non_self_params(f5)
    assert callable_has_non_self_params(f6)
    assert callable_has_non_self_params(f7)


def test_callable_has_non_self_non_default_params():
    assert not callable_has_non_self_non_default_params(Cls.f1)
    assert not callable_has_non_self_non_default_params(Cls().f1)
    assert callable_has_non_self_non_default_params(Cls.f2)
    assert callable_has_non_self_non_default_params(Cls().f2)
    assert callable_has_non_self_non_default_params(Cls.f3)
    assert callable_has_non_self_non_default_params(Cls().f3)
    assert not callable_has_non_self_non_default_params(Cls.f4)
    assert not callable_has_non_self_non_default_params(Cls().f4)
    assert not callable_has_non_self_non_default_params(f5)
    assert callable_has_non_self_non_default_params(f6)
    assert not callable_has_non_self_non_default_params(f7)


class Foo:
    def __init__(self):
        pass

    @method()
    def bar(self):
        return "hello"

    @web_endpoint()
    def web(self):
        pass


# run test on same event loop as servicer
@async_utils.synchronize_api
async def test_stream_function_call_data(servicer, client):
    req = api_pb2.FunctionCallPutDataRequest(
        function_call_id="fc-bar",
        data_chunks=[
            api_pb2.DataChunk(
                data_format=api_pb2.DATA_FORMAT_PICKLE,
                data=serialize_data_format("hello", api_pb2.DATA_FORMAT_PICKLE),
                index=1,
            ),
            api_pb2.DataChunk(
                data_format=api_pb2.DATA_FORMAT_PICKLE,
                data=serialize_data_format("world", api_pb2.DATA_FORMAT_PICKLE),
                index=2,
            ),
        ],
    )
    await client.stub.FunctionCallPutDataOut(req)

    t0 = time.time()
    gen = _stream_function_call_data(client, "fc-bar", "data_out")
    servicer.fail_get_data_out = [Status.INTERNAL] * 3
    assert await gen.__anext__() == "hello"
    elapsed = time.time() - t0
    assert 0.111 <= elapsed < 1.0

    assert await gen.__anext__() == "world"
