# Copyright Modal Labs 2022
import asyncio
from typing import Any, AsyncGenerator, Callable, Dict, NoReturn, Optional, cast

import aiohttp

from ._utils.async_utils import TaskContext
from ._utils.blob_utils import MAX_OBJECT_SIZE_BYTES
from ._utils.package_utils import parse_major_minor_version
from .config import logger
from .exception import ExecutionError, InvalidError
from .execution_context import current_function_call_id
from .experimental import stop_fetching_inputs

FIRST_MESSAGE_TIMEOUT_SECONDS = 5.0


def asgi_app_wrapper(asgi_app, function_io_manager) -> Callable[..., AsyncGenerator]:
    async def fn(scope):
        function_call_id = current_function_call_id()
        assert function_call_id, "internal error: function_call_id not set in asgi_app() scope"

        # TODO: Add support for the ASGI lifecycle spec.
        messages_from_app: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(1)
        messages_to_app: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(1)

        async def disconnect_app():
            if scope["type"] == "http":
                await messages_to_app.put({"type": "http.disconnect"})
            elif scope["type"] == "websocket":
                await messages_to_app.put({"type": "websocket.disconnect"})

        async def handle_first_input_timeout():
            if scope["type"] == "http":
                await messages_from_app.put({"type": "http.response.start", "status": 502})
                await messages_from_app.put(
                    {
                        "type": "http.response.body",
                        "body": b"Missing request, possibly due to expiry or cancellation",
                    }
                )
            elif scope["type"] == "websocket":
                await messages_from_app.put(
                    {
                        "type": "websocket.close",
                        "code": 1011,
                        "reason": "Missing request, possibly due to expiry or cancellation",
                    }
                )
            await disconnect_app()

        async def fetch_data_in():
            # Cancel an ASGI app call if the initial message is not received within a short timeout.
            #
            # This initial message, "http.request" or "websocket.connect", should be sent
            # immediately after starting the ASGI app's function call. If it is not received, that
            # indicates a request cancellation or other abnormal circumstance.
            message_gen = function_io_manager.get_data_in.aio(function_call_id)

            try:
                first_message = await asyncio.wait_for(message_gen.__anext__(), FIRST_MESSAGE_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, StopAsyncIteration):
                # About `StopAsyncIteration` above: The generator shouldn't typically exit,
                # but if it does, we handle it like a timeout in that case.
                await handle_first_input_timeout()
                return
            except Exception:
                logger.exception("Internal error in asgi_app_wrapper")
                await disconnect_app()
                return

            await messages_to_app.put(first_message)
            async for message in message_gen:
                await messages_to_app.put(message)

        async def send(msg):
            # Automatically split body chunks that are greater than the output size limit, to
            # prevent them from being uploaded to S3.
            if msg["type"] == "http.response.body":
                body_chunk_size = MAX_OBJECT_SIZE_BYTES - 1024  # reserve 1 KiB for framing
                body_chunk_limit = 20 * body_chunk_size
                s3_chunk_size = 50 * body_chunk_size

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
                        await messages_from_app.put({"type": "http.response.body", "body": chunk, "more_body": True})
                    msg["body"] = msg["body"][indices[-1] :]

            await messages_from_app.put(msg)

        # Run the ASGI app, while draining the send message queue at the same time,
        # and yielding results.
        async with TaskContext() as tc:
            tc.create_task(fetch_data_in())

            async def receive():
                return await messages_to_app.get()

            app_task = tc.create_task(asgi_app(scope, receive, send))
            pop_task = None
            while True:
                pop_task = tc.create_task(messages_from_app.get())

                try:
                    done, pending = await asyncio.wait([pop_task, app_task], return_when=asyncio.FIRST_COMPLETED)
                except asyncio.CancelledError:
                    break

                if pop_task in done:
                    yield pop_task.result()
                else:
                    # clean up the popping task, or we will leak unresolved tasks every loop iteration
                    pop_task.cancel()

                if app_task in done:
                    while not messages_from_app.empty():
                        yield messages_from_app.get_nowait()
                    app_task.result()  # consume/raise exceptions if there are any!
                    break

    return fn


def wsgi_app_wrapper(wsgi_app, function_io_manager):
    from ._vendor.a2wsgi_wsgi import WSGIMiddleware

    asgi_app = WSGIMiddleware(wsgi_app, workers=10000, send_queue_size=1)  # unlimited workers
    return asgi_app_wrapper(asgi_app, function_io_manager)


def webhook_asgi_app(fn: Callable, method: str, docs: bool):
    """Return a FastAPI app wrapping a function handler."""
    # Pulls in `fastapi` module, which is slow to import.
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(openapi_url="/openapi.json" if docs else None)  # disabling openapi spec disables all docs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_api_route("/", fn, methods=[method])
    return app


def get_ip_address(ifname: bytes):
    """Get the IP address associated with a network interface in Linux."""
    import fcntl
    import socket
    import struct

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(
        fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack("256s", ifname[:15]),
        )[20:24]
    )


def wait_for_web_server(host: str, port: int, *, timeout: float) -> None:
    """Wait until a web server port starts accepting TCP connections."""
    import socket
    import time

    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                break
        except OSError as ex:
            time.sleep(0.01)
            if time.monotonic() - start_time >= timeout:
                raise TimeoutError(
                    f"Waited too long for port {port} to start accepting connections. "
                    "Make sure the web server is bound to 0.0.0.0 (rather than localhost or 127.0.0.1), "
                    "or adjust `startup_timeout`."
                ) from ex


async def _proxy_http_request(session: aiohttp.ClientSession, scope, receive, send) -> None:
    proxy_response: aiohttp.ClientResponse

    async def request_generator() -> AsyncGenerator[bytes, None]:
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if body:
                    yield body
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                raise ConnectionAbortedError("Disconnect message received")
            else:
                raise ExecutionError(f"Unexpected message type: {message['type']}")

    path = scope["path"]
    if scope.get("query_string"):
        path += "?" + scope["query_string"].decode()

    try:
        proxy_response = await session.request(
            method=scope["method"],
            url=path,
            headers=[(k.decode(), v.decode()) for k, v in scope["headers"]],
            data=None if scope["method"] in aiohttp.ClientRequest.GET_METHODS else request_generator(),
            allow_redirects=False,
        )
    except ConnectionAbortedError:
        return
    except aiohttp.ClientConnectionError as e:  # some versions of aiohttp wrap the error
        if isinstance(e.__cause__, ConnectionAbortedError):
            return
        raise

    async def send_response() -> None:
        msg = {
            "type": "http.response.start",
            "status": proxy_response.status,
            "headers": [(k.encode(), v.encode()) for k, v in proxy_response.headers.items()],
        }
        await send(msg)
        async for data in proxy_response.content.iter_any():
            msg = {"type": "http.response.body", "body": data, "more_body": True}
            await send(msg)
        await send({"type": "http.response.body"})

    async def listen_for_disconnect() -> NoReturn:
        while True:
            message = await receive()
            if (
                message["type"] == "http.disconnect"
                and proxy_response.connection is not None
                and proxy_response.connection.transport is not None
            ):
                proxy_response.connection.transport.abort()

    async with TaskContext() as tc:
        send_response_task = tc.create_task(send_response())
        disconnect_task = tc.create_task(listen_for_disconnect())
        await asyncio.wait([send_response_task, disconnect_task], return_when=asyncio.FIRST_COMPLETED)


async def _proxy_websocket_request(session: aiohttp.ClientSession, scope, receive, send) -> None:
    first_message = await receive()  # Consume the initial "websocket.connect" message.
    if first_message["type"] == "websocket.disconnect":
        return
    elif first_message["type"] != "websocket.connect":
        raise ExecutionError(f"Unexpected message type: {first_message['type']}")

    path = scope["path"]
    if scope.get("query_string"):
        path += "?" + scope["query_string"].decode()

    async with session.ws_connect(
        url=path,
        headers=[(k.decode(), v.decode()) for k, v in scope["headers"]],  # type: ignore
        protocols=scope.get("subprotocols", []),
    ) as upstream_ws:

        async def client_to_upstream():
            while True:
                client_message = await receive()
                if client_message["type"] == "websocket.disconnect":
                    await upstream_ws.close(code=client_message.get("code", 1005))
                    break
                elif client_message["type"] == "websocket.receive":
                    if client_message.get("text") is not None:
                        await upstream_ws.send_str(client_message["text"])
                    elif client_message.get("bytes") is not None:
                        await upstream_ws.send_bytes(client_message["bytes"])
                else:
                    raise ExecutionError(f"Unexpected message type: {client_message['type']}")

        async def upstream_to_client():
            msg: Dict[str, Any] = {
                "type": "websocket.accept",
                "subprotocol": upstream_ws.protocol,
            }
            await send(msg)

            while True:
                upstream_message = await upstream_ws.receive()
                if upstream_message.type == aiohttp.WSMsgType.closed:
                    msg = {"type": "websocket.close"}
                    if upstream_message.data is not None:
                        msg["code"] = cast(aiohttp.WSCloseCode, upstream_message.data).value
                        msg["reason"] = upstream_message.extra
                    await send(msg)
                    break
                elif upstream_message.type == aiohttp.WSMsgType.text:
                    await send({"type": "websocket.send", "text": upstream_message.data})
                elif upstream_message.type == aiohttp.WSMsgType.binary:
                    await send({"type": "websocket.send", "bytes": upstream_message.data})
                else:
                    pass  # Ignore all other upstream WebSocket message types.

        async with TaskContext() as tc:
            client_to_upstream_task = tc.create_task(client_to_upstream())
            upstream_to_client_task = tc.create_task(upstream_to_client())
            await asyncio.wait([client_to_upstream_task, upstream_to_client_task], return_when=asyncio.FIRST_COMPLETED)


def web_server_proxy(host: str, port: int):
    """Return an ASGI app that proxies requests to a web server running on the same host."""
    if not 0 < port < 65536:
        raise InvalidError(f"Invalid port number: {port}")

    base_url = f"http://{host}:{port}"
    session: Optional[aiohttp.ClientSession] = None

    async def web_server_proxy_app(scope, receive, send):
        nonlocal session
        if session is None:
            # TODO: We currently create the ClientSession on container startup and never close it.
            # This outputs an "Unclosed client session" warning during runner termination. We should
            # properly close the session once we implement the ASGI lifespan protocol.
            session = aiohttp.ClientSession(
                base_url,
                cookie_jar=aiohttp.DummyCookieJar(),
                timeout=aiohttp.ClientTimeout(total=3600),
                auto_decompress=False,
                read_bufsize=1024 * 1024,  # 1 MiB
                **(
                    # These options were introduced in aiohttp 3.9, and we can remove the
                    # conditional after deprecating image builder version 2023.12.
                    dict(  # type: ignore
                        max_line_size=64 * 1024,  # 64 KiB
                        max_field_size=64 * 1024,  # 64 KiB
                    )
                    if parse_major_minor_version(aiohttp.__version__) >= (3, 9)
                    else {}
                ),
            )

        try:
            if scope["type"] == "lifespan":
                pass  # Do nothing for lifespan events.
            elif scope["type"] == "http":
                await _proxy_http_request(session, scope, receive, send)
            elif scope["type"] == "websocket":
                await _proxy_websocket_request(session, scope, receive, send)
            else:
                raise NotImplementedError(f"Scope {scope} is not understood")

        except aiohttp.ClientConnectorError as exc:
            # If the server is not running or not reachable, we should stop fetching new inputs.
            logger.warning(f"Terminating runner due to @web_server connection issue: {exc}")
            stop_fetching_inputs()

    return web_server_proxy_app
