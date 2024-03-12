# Copyright Modal Labs 2022
import asyncio
from typing import Any, AsyncGenerator, Callable, Dict, List

from ._utils.async_utils import TaskContext
from ._utils.blob_utils import MAX_OBJECT_SIZE_BYTES
from .config import logger
from .functions import current_function_call_id


class NoRequestReceived(Exception):
    pass


def asgi_app_wrapper(asgi_app, function_io_manager) -> Callable[..., AsyncGenerator]:
    async def fn(scope):
        function_call_id = current_function_call_id()
        assert function_call_id, "internal error: function_call_id not set in asgi_app() scope"

        # TODO: Add support for the ASGI lifecycle spec.
        messages_from_app: asyncio.Queue[List[Dict[str, Any]]] = asyncio.Queue(1)
        messages_to_app: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(1)

        async def fetch_data_in():
            # Cancel an ASGI app call if the initial message is not received within a short timeout.
            #
            # This initial message, "http.request" or "websocket.connect", should be sent
            # immediately after starting the ASGI app's function call. If it is not received, that
            # indicates a request cancellation or other abnormal circumstance.
            message_gen = function_io_manager.get_data_in.aio(function_call_id)
            try:
                first_message = await asyncio.wait_for(message_gen.__anext__(), 5.0)
            except (asyncio.TimeoutError, StopAsyncIteration):
                if scope["type"] == "http":
                    await messages_from_app.put(
                        [
                            {"type": "http.response.start", "status": 502},
                            {
                                "type": "http.response.body",
                                "body": b"Missing request, possibly due to cancellation or crash",
                            },
                        ]
                    )
                    await messages_to_app.put(
                        {"type": "http.disconnect"},
                    )
                elif scope["type"] == "websocket":
                    await messages_from_app.put(
                        [
                            {
                                "type": "websocket.close",
                                "code": 1011,
                                "reason": "Missing request, possibly due to cancellation or crash",
                            }
                        ]
                    )
                    await messages_to_app.put(
                        {"type": "websocket.disconnect"},
                    )
                else:
                    logger.error(f"scope is neither http nor websocket {scope['type']})")
                raise NoRequestReceived()

            await messages_to_app.put(first_message)
            async for message in message_gen:
                await messages_to_app.put(message)

        async def send(msg):
            # Automatically split body chunks that are greater than the output size limit, to
            # prevent them from being uploaded to S3.
            if msg["type"] == "http.response.body":
                body_chunk_size = MAX_OBJECT_SIZE_BYTES - 1024  # reserve 1 KiB for framing
                body_chunk_limit = 20 * body_chunk_size
                s3_chunk_size = 150 * body_chunk_size

                size = len(msg.get("body", b""))
                if size <= body_chunk_limit:
                    chunk_size = body_chunk_size
                else:
                    # If the body is _very large_, we should still split it up to avoid sending all
                    # of the data in a huge chunk in S3.
                    chunk_size = s3_chunk_size

                if size > chunk_size:
                    indices = list(range(0, size, chunk_size))
                    for i in indices[:-1]:
                        chunk = msg["body"][i : i + chunk_size]
                        await messages_from_app.put([{"type": "http.response.body", "body": chunk, "more_body": True}])
                    msg["body"] = msg["body"][indices[-1] :]

            await messages_from_app.put([msg])

        async def receive():
            return await messages_to_app.get()

        # Run the ASGI app, while draining the send message queue at the same time,
        # and yielding results.
        async with TaskContext(grace=0.01) as tc:
            app_task = tc.create_task(asgi_app(scope, receive, send))
            fetch_data_in_task = tc.create_task(fetch_data_in())
            pop_task = None
            try:
                while True:
                    pop_task = tc.create_task(messages_from_app.get())

                    try:
                        done, pending = await asyncio.wait(
                            [pop_task, app_task, fetch_data_in_task], return_when=asyncio.FIRST_COMPLETED
                        )
                    except asyncio.CancelledError:
                        break

                    if pop_task in done:
                        res = pop_task.result()
                        for msg in res:
                            yield msg
                    else:
                        pop_task.cancel()  # clean up the popping task, or we will leak unresolved tasks every loop iteration

                    if app_task in done:
                        while not messages_from_app.empty():
                            res = messages_from_app.get_nowait()
                            for msg in res:
                                yield msg

                        app_task.result()  # consume/raise exceptions if there are any!
                        break

                if fetch_data_in_task.done():
                    # the data fetching loop task should typically not exit, not even gracefully
                    logger.error(f"Data fetching task stopped unexpectedly: {fetch_data_in_task.exception()}")
                    fetch_data_in_task.result()  # in case there were exceptions, raise those
            finally:
                fetch_data_in_task.cancel()
                if not app_task.done():
                    # only cancel in case it's not done - this lets tracebacks from potential errors
                    # still get displayed when the task gets garbage collected
                    app_task.cancel()
                if not pop_task.done():
                    # only cancel in case it's not done - this lets tracebacks from potential errors
                    # still get displayed when the task gets garbage collected
                    pop_task.cancel()

    return fn


def wsgi_app_wrapper(wsgi_app, function_io_manager):
    from ._vendor.a2wsgi_wsgi import WSGIMiddleware

    asgi_app = WSGIMiddleware(wsgi_app, workers=10000, send_queue_size=1)  # unlimited workers
    return asgi_app_wrapper(asgi_app, function_io_manager)


def webhook_asgi_app(fn: Callable, method: str):
    """Return a FastAPI app wrapping a function handler."""
    # Pulls in `fastapi` module, which is slow to import.
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_api_route("/", fn, methods=[method])
    return app
