# Copyright Modal Labs 2024
import asyncio
import pytest

import fastapi
from starlette.requests import ClientDisconnect

from modal._asgi import asgi_app_wrapper
from modal.execution_context import _set_current_context_ids


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
    body = await req.body()
    return {"some_result": body}


@app.get("/async_error")
async def async_error():
    raise DummyException()


@app.get("/streaming_response")
async def streaming_response():
    from fastapi.responses import StreamingResponse

    async def stream_bytes():
        yield b"foo"
        yield b"bar"

    return StreamingResponse(stream_bytes())


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
@pytest.mark.timeout(1)
async def test_success():
    mock_manager = MockIOManager()
    _set_current_context_ids(["in-123"], ["fc-123"])
    wrapped_app, lifespan_manager = asgi_app_wrapper(app, mock_manager)
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
@pytest.mark.timeout(1)
async def test_endpoint_exception(endpoint_url):
    mock_manager = MockIOManager()
    _set_current_context_ids(["in-123"], ["fc-123"])
    wrapped_app, lifespan_manager = asgi_app_wrapper(app, mock_manager)
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


class BrokenIOManager:
    class get_data_in:
        @staticmethod
        async def aio(_function_call_id):
            raise DummyException("error while fetching data")
            yield  # noqa (makes this a generator)


@pytest.mark.asyncio
@pytest.mark.timeout(1)
async def test_broken_io_unused(caplog):
    # if IO channel breaks, but the endpoint doesn't actually use
    # any of the body data, it should be allowed to output its data
    # and not raise an exception - but print a warning since it's unexpected
    mock_manager = BrokenIOManager()
    _set_current_context_ids(["in-123"], ["fc-123"])
    wrapped_app, lifespan_manager = asgi_app_wrapper(app, mock_manager)
    asgi_scope = _asgi_get_scope("/")
    outputs = []

    async for output in wrapped_app(asgi_scope):
        outputs.append(output)

    assert len(outputs) == 2
    assert outputs[0]["status"] == 200
    assert outputs[1]["body"] == b'{"some_result":"foo"}'
    assert "Internal error" in caplog.text
    assert "DummyException: error while fetching data" in caplog.text


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_broken_io_used():
    mock_manager = BrokenIOManager()
    _set_current_context_ids(["in-123"], ["fc-123"])
    wrapped_app, lifespan_manager = asgi_app_wrapper(app, mock_manager)
    asgi_scope = _asgi_get_scope("/async_reading_body", "POST")
    outputs = []
    with pytest.raises(ClientDisconnect):
        async for output in wrapped_app(asgi_scope):
            outputs.append(output)

    assert len(outputs) == 2
    assert outputs[0]["status"] == 500


class SlowIOManager:
    class get_data_in:
        @staticmethod
        async def aio(_function_call_id):
            await asyncio.sleep(5)
            yield  # makes this an async generator


@pytest.mark.asyncio
@pytest.mark.timeout(2)
async def test_first_message_timeout(monkeypatch):
    monkeypatch.setattr("modal._asgi.FIRST_MESSAGE_TIMEOUT_SECONDS", 0.1)  # simulate timeout
    _set_current_context_ids(["in-123"], ["fc-123"])
    wrapped_app, lifespan_manager = asgi_app_wrapper(app, SlowIOManager())
    asgi_scope = _asgi_get_scope("/async_reading_body", "POST")
    outputs = []
    with pytest.raises(ClientDisconnect):
        async for output in wrapped_app(asgi_scope):
            outputs.append(output)

    assert outputs[0]["status"] == 502
    assert b"Missing request" in outputs[1]["body"]


@pytest.mark.asyncio
async def test_cancellation_cleanup(caplog):
    # this test mostly exists to get some coverage on the cancellation/error paths and
    # ensure nothing unexpected happens there
    _set_current_context_ids(["in-123"], ["fc-123"])
    wrapped_app, lifespan_manager = asgi_app_wrapper(app, SlowIOManager())
    asgi_scope = _asgi_get_scope("/async_reading_body", "POST")
    outputs = []

    async def app_runner():
        async for output in wrapped_app(asgi_scope):
            outputs.append(output)

    app_runner_task = asyncio.create_task(app_runner())
    await asyncio.sleep(0.1)  # let it get started
    app_runner_task.cancel()
    await asyncio.sleep(0.1)  # let it shut down
    assert len(outputs) == 0
    assert caplog.text == ""  # make sure there are no junk traces about dangling tasks etc.


@pytest.mark.asyncio
async def test_streaming_response():
    _set_current_context_ids(["in-123"], ["fc-123"])
    wrapped_app, lifespan_manager = asgi_app_wrapper(app, SlowIOManager())
    asgi_scope = _asgi_get_scope("/streaming_response", "GET")
    outputs = []
    async for output in wrapped_app(asgi_scope):
        outputs.append(output)
    assert outputs == [
        {"headers": [], "status": 200, "type": "http.response.start"},
        {"body": b"foo", "more_body": True, "type": "http.response.body"},
        {"body": b"bar", "more_body": True, "type": "http.response.body"},
        {"body": b"", "more_body": False, "type": "http.response.body"},
    ]


class StreamingIOManager:
    class get_data_in:
        @staticmethod
        async def aio(_function_call_id):
            yield {"type": "http.request", "body": b"foo", "more_body": True}
            yield {"type": "http.request", "body": b"bar", "more_body": True}
            yield {"type": "http.request", "body": b"baz", "more_body": False}
            yield {"type": "http.request", "body": b"this should not be read", "more_body": False}


@pytest.mark.asyncio
async def test_streaming_body():
    _set_current_context_ids(["in-123"], ["fc-123"])

    wrapped_app, lifespan_manager = asgi_app_wrapper(app, StreamingIOManager())
    asgi_scope = _asgi_get_scope("/async_reading_body", "POST")
    outputs = []
    async for output in wrapped_app(asgi_scope):
        outputs.append(output)
    assert outputs[1] == {"type": "http.response.body", "body": b'{"some_result":"foobarbaz"}'}


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_first_input():
    # due to an asyncio edge case of cancellation + wait_for(future) resolution there
    # are scenarios in which an asgi task cancellation doesn't actually stop the underlying
    # fetch_data_in task, causing either warnings on shutdown or even infinite stalling on
    # shutdown.
    _set_current_context_ids(["in-123"], ["fc-123"])
    fut: asyncio.Future[None] = asyncio.Future()

    class StreamingIOManager:
        class get_data_in:
            @staticmethod
            async def aio(_function_call_id):
                await fut  # we never resolve this, unlike in test_cancellation_first_message_race_cleanup
                yield

    wrapped_app, _ = asgi_app_wrapper(app, StreamingIOManager())
    asgi_scope = _asgi_get_scope("/async_reading_body", "POST")

    first_app_output = asyncio.create_task(wrapped_app(asgi_scope).__anext__())  # type: ignore
    await asyncio.sleep(0.1)  # ensure we are in wait_for(first_message_task)
    first_app_output.cancel()
    await asyncio.sleep(0.1)  # resume event loop to resolve tasks if possible
    remaining_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert len(remaining_tasks) == 0


@pytest.mark.asyncio
async def test_cancellation_when_first_input_arrives():
    # due to an asyncio edge case of cancellation + wait_for(future) resolution there
    # are scenarios in which an asgi task cancellation doesn't actually stop the underlying
    # fetch_data_in task, causing either warnings on shutdown or even infinite stalling on
    # shutdown.
    _set_current_context_ids(["in-123"], ["fc-123"])
    fut: asyncio.Future[None] = asyncio.Future()

    class StreamingIOManager:
        class get_data_in:
            @staticmethod
            async def aio(_function_call_id):
                await fut
                yield {"type": "http.request", "body": b"foo", "more_body": True}
                while 1:
                    yield  # simulate infinite stream

    wrapped_app, _ = asgi_app_wrapper(app, StreamingIOManager())
    asgi_scope = _asgi_get_scope("/async_reading_body", "POST")

    first_app_output = asyncio.create_task(wrapped_app(asgi_scope).__anext__())  # type: ignore
    await asyncio.sleep(0.1)  # ensure we are in wait_for(first_message_task)
    # now lets unblock get_data_in, supplying a request to the waiting asgi app
    # fut.set_result(None)
    # but at the same time, before we resume the event loop, we cancel the full input task
    fut.set_result(None)
    first_app_output.cancel()
    await asyncio.sleep(0.1)  # resume event loop to resolve tasks if possible
    remaining_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert len(remaining_tasks) == 0
