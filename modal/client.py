# Copyright Modal Labs 2022
import asyncio
import platform
import warnings
from typing import Awaitable, Callable, Dict, Optional, Tuple

from aiohttp import ClientConnectorError, ClientResponseError
from google.protobuf import empty_pb2
from grpclib import GRPCError, Status

from modal_proto import api_grpc, api_pb2
from modal_utils import async_utils
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import create_channel, retry_transient_errors
from modal_utils.http_utils import http_client_with_tls
from modal_version import __version__

from ._tracing import inject_tracing_context
from .config import config, logger
from .exception import AuthError, ConnectionError, DeprecationError, VersionError

HEARTBEAT_INTERVAL: float = config.get("heartbeat_interval")
HEARTBEAT_TIMEOUT: float = 10.1
CLIENT_CREATE_ATTEMPT_TIMEOUT: float = 4.0
CLIENT_CREATE_TOTAL_TIMEOUT: float = 15.0


def _get_metadata(client_type: int, credentials: Optional[Tuple[str, str]], version: str) -> Dict[str, str]:
    # This implements a simplified version of platform.platform() that's still machine-readable
    uname: platform.uname_result = platform.uname()
    if uname.system == "Darwin":
        system, release = "macOS", platform.mac_ver()[0]
    else:
        system, release = uname.system, uname.release
    platform_str = "-".join(s.replace("-", "_") for s in (system, release, uname.machine))

    metadata = {
        "x-modal-client-version": version,
        "x-modal-client-type": str(client_type),
        "x-modal-python-version": platform.python_version(),
        "x-modal-node": platform.node(),
        "x-modal-platform": platform_str,
    }
    if credentials and client_type == api_pb2.CLIENT_TYPE_CLIENT:
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

    client_type: int

    def __init__(
        self,
        server_url,
        client_type,
        credentials,
        version=__version__,
        *,
        no_verify=False,
    ):
        self.server_url = server_url
        self.client_type = client_type
        self.credentials = credentials
        self.version = version
        self.no_verify = no_verify
        self._pre_stop: Optional[Callable[[], Awaitable[None]]] = None
        self._channel = None
        self._stub: Optional[api_grpc.ModalClientStub] = None

    @property
    def stub(self) -> Optional[api_grpc.ModalClientStub]:
        """mdmd:hidden"""
        return self._stub

    async def _open(self):
        assert self._stub is None
        metadata = _get_metadata(self.client_type, self.credentials, self.version)
        self._channel = create_channel(
            self.server_url,
            metadata=metadata,
            inject_tracing_context=inject_tracing_context,
        )
        self._stub = api_grpc.ModalClientStub(self._channel)  # type: ignore

    async def _close(self):
        if self._pre_stop is not None:
            logger.debug("Client: running pre-stop coroutine before shutting down")
            await self._pre_stop()  # type: ignore

        if self._channel is not None:
            self._channel.close()

        # Remove cached client.
        self.set_env_client(None)

    def set_pre_stop(self, pre_stop: Callable[[], Awaitable[None]]):
        """mdmd:hidden"""
        # hack: stub.serve() gets into a losing race with the `on_shutdown` client
        # teardown when an interrupt signal is received (eg. KeyboardInterrupt).
        # By registering a pre-stop fn stub.serve() can have its teardown
        # performed before the client is disconnected.
        #
        # ref: github.com/modal-labs/modal-client/pull/108
        self._pre_stop = pre_stop

    async def _verify(self):
        logger.debug("Client: Starting")
        try:
            req = empty_pb2.Empty()
            resp = await retry_transient_errors(
                self.stub.ClientHello,
                req,
                attempt_timeout=CLIENT_CREATE_ATTEMPT_TIMEOUT,
                total_timeout=CLIENT_CREATE_TOTAL_TIMEOUT,
            )
            if resp.warning:
                ALARM_EMOJI = chr(0x1F6A8)
                warnings.warn(f"{ALARM_EMOJI} {resp.warning} {ALARM_EMOJI}", DeprecationError)
        except GRPCError as exc:
            if exc.status == Status.FAILED_PRECONDITION:
                raise VersionError(
                    f"The client version {self.version} is too old. Please update to the latest package on PyPi: https://pypi.org/project/modal"
                )
            elif exc.status == Status.UNAUTHENTICATED:
                raise AuthError(exc.message)
            else:
                exc_string = await _grpc_exc_string(exc, "ClientHello", self.server_url, CLIENT_CREATE_TOTAL_TIMEOUT)
                raise ConnectionError(exc_string)
        except (OSError, asyncio.TimeoutError) as exc:
            raise ConnectionError(str(exc))

    async def __aenter__(self):
        await self._open()
        if not self.no_verify:
            await self._verify()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._close()

    @classmethod
    async def verify(cls, server_url, credentials):
        """mdmd:hidden"""
        async with _Client(server_url, api_pb2.CLIENT_TYPE_CLIENT, credentials):
            pass  # Will call ClientHello

    @classmethod
    async def unauthenticated_client(cls, server_url: str):
        """mdmd:hidden"""
        # Create a connection with no credentials
        # To be used with the token flow
        return _Client(server_url, api_pb2.CLIENT_TYPE_CLIENT, None, no_verify=True)

    @classmethod
    async def from_env(cls, _override_config=None) -> "_Client":
        """mdmd:hidden"""
        if _override_config:
            # Only used for testing
            c = _override_config
        else:
            c = config

        # Sets server_url to socket file path if proxy is available.
        server_url = c["server_url"]

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
                await client._open()
                async_utils.on_shutdown(client._close())
                try:
                    await client._verify()
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
                return client

    @classmethod
    def set_env_client(cls, client: Optional["_Client"]):
        """mdmd:hidden"""
        # Just used from tests.
        cls._client_from_env = client


Client = synchronize_api(_Client)
