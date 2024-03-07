import asyncio
import pytest
import time

import fastapi

from modal._asgi import asgi_app_wrapper
from modal.functions import _set_current_context_ids


class DummyException(Exception):
    pass


app = fastapi.FastAPI()


@app.get("/")
def sync_index():
    return {"some_result": "foo"}


@app.get("/error")
def sync_error():
    raise DummyException()


@app.post("/async_reading_body")
async def async_index_reading_body(req: fastapi.Request):
    await req.body()
    return {"some_result": "foo"}


@app.get("/async_error")
async def async_error():
    raise DummyException()


@app.get("/async_slow")
async def async_slow():
    await asyncio.sleep(1)


@app.get("/slow")
def slow():
    # TODO: a *sync* task like this will actually keep in fast apis own worker threads,
    #  even if we cancel the asgi call. Hard to fix, since it would both require
    #  changing fastapi internals, and cancelling user user code in threads
    time.sleep(1)


def _asgi_get_scope(path, method="GET"):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": "",
        "headers": [],
    }


class MockIOManager:
    class get_data_in:
        @staticmethod
        async def aio(_function_call_id):
            yield {"type": "http.request", "body": b"some_body"}
            await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_success():
    mock_manager = MockIOManager()
    _set_current_context_ids("in-123", "fc-123")
    wrapped_app = asgi_app_wrapper(app, mock_manager)
    asgi_scope = _asgi_get_scope("/")
    outputs = [output async for output in wrapped_app(asgi_scope)]
    assert len(outputs) == 2
    before_body = outputs[0]
    assert before_body["status"] == 200
    assert before_body["type"] == "http.response.start"
    body = outputs[1]
    assert body["body"] == b'{"some_result":"foo"}'
    assert body["type"] == "http.response.body"


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_url", ["/error", "/async_error"])
async def test_endpoint_exception(endpoint_url):
    mock_manager = MockIOManager()
    _set_current_context_ids("in-123", "fc-123")
    wrapped_app = asgi_app_wrapper(app, mock_manager)
    asgi_scope = _asgi_get_scope(endpoint_url)
    outputs = []

    with pytest.raises(DummyException):
        async for output in wrapped_app(asgi_scope):
            outputs.append(output)

    assert len(outputs) == 2
    before_body = outputs[0]
    assert before_body["status"] == 500
    assert before_body["type"] == "http.response.start"
    body = outputs[1]
    assert body["body"] == b"Internal Server Error"
    assert body["type"] == "http.response.body"


@pytest.mark.asyncio
async def test_broken_io_unused(caplog):
    # if IO channel breaks, but the endpoint doesn't actually use
    # any of the body data, it should be allowed to output its data
    # to the channel before we raise the relevant exception
    class BrokenIOManager:
        class get_data_in:
            @staticmethod
            async def aio(_function_call_id):
                raise DummyException("error while fetching data")
                yield  # noqa

    mock_manager = BrokenIOManager()
    _set_current_context_ids("in-123", "fc-123")
    wrapped_app = asgi_app_wrapper(app, mock_manager)
    asgi_scope = _asgi_get_scope("/")
    outputs = []

    with pytest.raises(DummyException, match="error while fetching data"):
        async for output in wrapped_app(asgi_scope):
            outputs.append(output)

    assert len(outputs) == 2
    assert outputs[0]["status"] == 200
    assert outputs[1]["body"] == b'{"some_result":"foo"}'
    assert "Data fetching task stopped unexpectedly" in caplog.text


@pytest.mark.asyncio
async def test_app_reads_broken_data():
    class NoDataIOManager:
        class get_data_in:
            @staticmethod
            async def aio(_function_call_id):
                yield {}  # this asgi message has no "type"

    mock_manager = NoDataIOManager()
    _set_current_context_ids("in-123", "fc-123")
    wrapped_app = asgi_app_wrapper(app, mock_manager)
    asgi_scope = _asgi_get_scope("/async_reading_body", "POST")
    outputs = []
    with pytest.raises(KeyError, match="type"):
        async for output in wrapped_app(asgi_scope):
            outputs.append(output)

    assert len(outputs) == 2
    assert outputs[0]["status"] == 500


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_url", ["/slow", "/async_slow"])
async def test_cancel_asgi_task_before_output(endpoint_url):
    mock_manager = MockIOManager()
    _set_current_context_ids("in-123", "fc-123")
    wrapped_app = asgi_app_wrapper(app, mock_manager)
    asgi_scope = _asgi_get_scope(endpoint_url)
    outputs = []
    t0 = time.monotonic()
    async with asyncio.timeout(0.1):  # this should cancel the endpoint
        async for output in wrapped_app(asgi_scope):
            outputs.append(output)
    assert 0.1 < time.monotonic() - t0 < 0.5
    assert not outputs  # no outputs emitted when we cancel the call
    assert len(asyncio.all_tasks()) == 1  # only this test should be an active task
