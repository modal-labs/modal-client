# Copyright Modal Labs 2022
from __future__ import annotations

import asyncio
import platform
import warnings
import webbrowser
from typing import Callable, Optional

from aiohttp import ClientConnectorError, ClientResponseError
from grpclib import GRPCError, Status
from rich.console import Console

from modal_proto import api_grpc, api_pb2
from modal_utils import async_utils
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import (
    create_channel,
    retry_transient_errors,
)
from modal_utils.http_utils import http_client_with_tls
from modal_version import __version__

from ._tracing import inject_tracing_context
from .config import config, logger
from .exception import (
    AuthError,
    ConnectionError,
    DeprecationError,
    InvalidError,
    VersionError,
)

HEARTBEAT_INTERVAL = 15.0
HEARTBEAT_TIMEOUT = 10.1
CLIENT_CREATE_ATTEMPT_TIMEOUT = 4.0
CLIENT_CREATE_TOTAL_TIMEOUT = 15.0


def _get_metadata(client_type: int, credentials: Optional[tuple[str, str]], version: str) -> dict[str, str]:
    metadata = {
        "x-modal-client-version": version,
        "x-modal-client-type": str(client_type),
    }
    if credentials and (client_type == api_pb2.CLIENT_TYPE_CLIENT or client_type == api_pb2.CLIENT_TYPE_WEB_SERVER):
        token_id, token_secret = credentials
        metadata.update(
            {
                "x-modal-token-id": token_id,
                "x-modal-token-secret": token_secret,
            }
        )
    elif credentials and client_type == api_pb2.CLIENT_TYPE_CONTAINER:
        task_id, task_secret = credentials
        metadata.update(
            {
                "x-modal-task-id": task_id,
                "x-modal-task-secret": task_secret,
            }
        )
    return metadata


async def _http_check(url: str, timeout: float) -> str:
    # Used for sanity checking connection issues
    try:
        async with http_client_with_tls(timeout=timeout) as session:
            async with session.get(url) as resp:
                return f"HTTP status: {resp.status}"
    except ClientResponseError as exc:
        return f"HTTP status: {exc.status}"
    except ClientConnectorError as exc:
        return f"HTTP exception: {exc.os_error.__class__.__name__}"
    except Exception as exc:
        return f"HTTP exception: {exc.__class__.__name__}"


async def _grpc_exc_string(exc: GRPCError, method_name: str, server_url: str, timeout: float) -> str:
    http_status = await _http_check(server_url, timeout=timeout)
    return f"{method_name}: {exc.message} [GRPC status: {exc.status.name}, {http_status}]"


class _Client:
    _client_from_env = None
    _client_from_env_lock = None

    def __init__(
        self,
        server_url,
        client_type,
        credentials,
        version=__version__,
    ):
        self.server_url = server_url
        self.client_type = client_type
        self.credentials = credentials
        self.version = version
        self._stub = None
        self._connected = False
        self._pre_stop: Optional[Callable[[], None]] = None
        self._channel = None

    @property
    def stub(self):
        if self._stub is None:
            raise ConnectionError("The client is not connected to the modal server")
        return self._stub

    async def _start(self):
        logger.debug("Client: Starting")
        if self._stub:
            raise Exception("Client is already running")
        metadata = _get_metadata(self.client_type, self.credentials, self.version)
        self._channel = create_channel(
            self.server_url,
            metadata=metadata,
            inject_tracing_context=inject_tracing_context,
        )
        self._stub = api_grpc.ModalClientStub(self._channel)  # type: ignore
        try:
            req = api_pb2.ClientCreateRequest(
                client_type=self.client_type,
                version=self.version,
            )
            resp = await retry_transient_errors(
                self.stub.ClientCreate,
                req,
                attempt_timeout=CLIENT_CREATE_ATTEMPT_TIMEOUT,
                total_timeout=CLIENT_CREATE_TOTAL_TIMEOUT,
            )
            if resp.deprecation_warning:
                ALARM_EMOJI = chr(0x1F6A8)
                warnings.warn(f"{ALARM_EMOJI} {resp.deprecation_warning} {ALARM_EMOJI}", DeprecationError)
            if not resp.client_id:
                raise InvalidError("Did not get a client id from server")
            self._client_id = resp.client_id
            self._connected = True
        except GRPCError as exc:
            if exc.status == Status.FAILED_PRECONDITION:
                raise VersionError(
                    f"The client version {self.version} is too old. Please update to the latest package on PyPi: https://pypi.org/project/modal-client"
                )
            elif exc.status == Status.UNAUTHENTICATED:
                raise AuthError(exc.message)
            else:
                exc_string = await _grpc_exc_string(exc, "ClientCreate", self.server_url, CLIENT_CREATE_TOTAL_TIMEOUT)
                raise ConnectionError(exc_string)
        except (OSError, asyncio.TimeoutError) as exc:
            raise ConnectionError(str(exc))
        finally:
            if not self._connected:
                # Tear down the channel pool etc
                await self._stop()

        logger.debug("Client: Done starting")

    def set_pre_stop(self, pre_stop: Callable[[], None]):
        """mdmd:hidden"""
        # hack: stub.serve() gets into a losing race with the `on_shutdown` client
        # teardown when an interrupt signal is received (eg. KeyboardInterrupt).
        # By registering a pre-stop fn stub.serve() can have its teardown
        # performed before the client is disconnected.
        #
        # ref: github.com/modal-labs/modal-client/pull/108
        self._pre_stop = pre_stop

    async def _stop(self):
        if self._pre_stop:
            logger.debug("Client: running pre-stop coroutine before shutting down")
            await self._pre_stop()  # type: ignore
        # TODO: we should trigger this using an exit handler
        logger.debug("Client: Shutting down")
        self._stub = None  # prevent any additional calls
        if self._channel:
            self._channel.close()
            self._channel = None
        logger.debug("Client: Done shutting down")
        # Needed to catch straggling CancelledErrors and GeneratorExits that propagate
        # through our chains of async generators.
        await asyncio.sleep(0.01)

    async def __aenter__(self):
        try:
            await self._start()
        except BaseException:
            await self._stop()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._stop()

    async def verify(self):
        async with self:
            # Just connect and disconnect
            pass

    @property
    def client_id(self):
        """A unique identifier for the Client."""
        return self._client_id

    @classmethod
    async def token_flow(cls, env: str, server_url: str):
        """Gets a token through a web flow."""

        # Create a connection with no credentials
        metadata = _get_metadata(api_pb2.CLIENT_TYPE_CLIENT, None, __version__)
        channel = create_channel(server_url, metadata)
        stub = api_grpc.ModalClientStub(channel)  # type: ignore

        try:
            # Create token creation request
            # Send some strings identifying the computer (these are shown to the user for security reasons)
            create_req = api_pb2.TokenFlowCreateRequest(
                node_name=platform.node(),
                platform_name=platform.platform(),
            )
            create_resp = await stub.TokenFlowCreate(create_req)

            console = Console()
            with console.status("Waiting for authentication in the web browser...", spinner="dots"):
                # Open the web url in the browser
                link_text = f"[link={create_resp.web_url}]{create_resp.web_url}[/link]"
                console.print(f"Launching {link_text} in your browser window")
                if webbrowser.open_new_tab(create_resp.web_url):
                    console.print("If this is not showing up, please copy the URL into your web browser manually")
                else:
                    console.print(
                        "[red]Was not able to launch web browser[/red]"
                        " - please go to the URL manually and complete the flow"
                    )

                # Wait for token forever
                while True:
                    wait_req = api_pb2.TokenFlowWaitRequest(token_flow_id=create_resp.token_flow_id, timeout=15.0)
                    wait_resp = await stub.TokenFlowWait(wait_req)
                    if not wait_resp.timeout:
                        console.print("[green]Success![/green]")
                        return (wait_resp.token_id, wait_resp.token_secret)
        finally:
            channel.close()

    @classmethod
    async def from_env(cls, _override_config=None) -> "_Client":
        if _override_config:
            # Only used for testing
            c = _override_config
        else:
            c = config

        # Sets server_url to socket file path if proxy is available.
        server_url = c["server_url"]
        if c.get("worker_proxy_active"):
            server_url = c["server_socket_filename"]

        token_id = c["token_id"]
        token_secret = c["token_secret"]
        task_id = c["task_id"]
        task_secret = c["task_secret"]

        if task_id and task_secret:
            client_type = api_pb2.CLIENT_TYPE_CONTAINER
            credentials = (task_id, task_secret)
        elif token_id and token_secret:
            client_type = api_pb2.CLIENT_TYPE_CLIENT
            credentials = (token_id, token_secret)
        else:
            client_type = api_pb2.CLIENT_TYPE_CLIENT
            credentials = None

        if cls._client_from_env_lock is None:
            cls._client_from_env_lock = asyncio.Lock()

        async with cls._client_from_env_lock:
            if cls._client_from_env:
                return cls._client_from_env
            else:
                client = _Client(server_url, client_type, credentials)
                try:
                    await client._start()
                except AuthError:
                    if not credentials:
                        creds_missing_msg = (
                            "Token missing. Could not authenticate client. "
                            "If you have token credentials, see modal.com/docs/reference/modal.config for setup help. "
                            "If you are a new user, register an account at modal.com, then run `modal token new`."
                        )
                        raise AuthError(creds_missing_msg)
                    else:
                        raise
                cls._client_from_env = client
                async_utils.on_shutdown(AioClient.stop_env_client())
                return client

    @classmethod
    def set_env_client(cls, client):
        """Just used from tests."""
        cls._client_from_env = client

    @classmethod
    async def stop_env_client(cls):
        # Only called from atexit handler and from tests
        if cls._client_from_env is not None:
            await cls._client_from_env._stop()
            cls._client_from_env = None


Client, AioClient = synchronize_apis(_Client)
