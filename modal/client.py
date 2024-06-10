# Copyright Modal Labs 2022
import asyncio
import platform
import warnings
from typing import AsyncIterator, Awaitable, Callable, ClassVar, Dict, Optional, Tuple

import grpclib.client
from aiohttp import ClientConnectorError, ClientResponseError
from google.protobuf import empty_pb2
from grpclib import GRPCError, Status
from synchronicity.async_wrap import asynccontextmanager

from modal_proto import api_grpc, api_pb2
from modal_version import __version__

from ._utils import async_utils
from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import create_channel, retry_transient_errors
from ._utils.http_utils import http_client_with_tls
from .config import _check_config, config, logger
from .exception import AuthError, ConnectionError, DeprecationError, VersionError

HEARTBEAT_INTERVAL: float = config.get("heartbeat_interval")
HEARTBEAT_TIMEOUT: float = HEARTBEAT_INTERVAL + 0.1
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
    return f"{method_name}: {exc.message} [gRPC status: {exc.status.name}, {http_status}]"


class _Client:
    _client_from_env: ClassVar[Optional["_Client"]] = None
    _client_from_env_lock: ClassVar[Optional[asyncio.Lock]] = None

    def __init__(
        self,
        server_url: str,
        client_type: int,
        credentials: Optional[Tuple[str, str]],
        version: str = __version__,
    ):
        """The Modal client object is not intended to be instantiated directly by users."""
        self.server_url = server_url
        self.client_type = client_type
        self._credentials = credentials
        self.version = version
        self._authenticated = False
        self.image_builder_version: Optional[str] = None
        self._pre_stop: Optional[Callable[[], Awaitable[None]]] = None
        self._channel: Optional[grpclib.client.Channel] = None
        self._stub: Optional[api_grpc.ModalClientStub] = None

    @property
    def stub(self) -> Optional[api_grpc.ModalClientStub]:
        """mdmd:hidden"""
        return self._stub

    @property
    def authenticated(self) -> bool:
        """mdmd:hidden"""
        return self._authenticated

    @property
    def credentials(self) -> tuple:
        """mdmd:hidden"""
        if self._credentials is None and self.client_type == api_pb2.CLIENT_TYPE_CONTAINER:
            logger.debug("restoring credentials for memory snapshotted client instance")
            self._credentials = (config["task_id"], config["task_secret"])
        return self._credentials

    async def _open(self):
        assert self._stub is None
        metadata = _get_metadata(self.client_type, self._credentials, self.version)
        self._channel = create_channel(self.server_url, metadata=metadata)
        self._stub = api_grpc.ModalClientStub(self._channel)  # type: ignore

    async def _close(self, forget_credentials: bool = False):
        if self._pre_stop is not None:
            logger.debug("Client: running pre-stop coroutine before shutting down")
            await self._pre_stop()  # type: ignore

        if self._channel is not None:
            self._channel.close()

        if forget_credentials:
            self._credentials = None

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

    async def _init(self):
        """Connect to server and retrieve version information; raise appropriate error for various failures."""
        logger.debug("Client: Starting")
        _check_config()
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
            self._authenticated = True
            self.image_builder_version = resp.image_builder_version
        except GRPCError as exc:
            if exc.status == Status.FAILED_PRECONDITION:
                raise VersionError(
                    f"The client version ({self.version}) is too old. Please update (pip install --upgrade modal)."
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
        try:
            await self._init()
        except BaseException:
            await self._close()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._close()

    @classmethod
    @asynccontextmanager
    async def anonymous(cls, server_url: str) -> AsyncIterator["_Client"]:
        """mdmd:hidden
        Create a connection with no credentials; to be used for token creation.
        """
        logger.debug("Client: Starting client without authentication")
        client = cls(server_url, api_pb2.CLIENT_TYPE_CLIENT, credentials=None)
        try:
            await client._open()
            # Skip client._init
            yield client
        finally:
            await client._close()

    @classmethod
    async def from_env(cls, _override_config=None) -> "_Client":
        """mdmd:hidden
        Singleton that is instantiated from the Modal config and reused on subsequent calls.
        """
        if _override_config:
            # Only used for testing
            c = _override_config
        else:
            c = config

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
                    await client._init()
                except AuthError:
                    if not credentials:
                        creds_missing_msg = (
                            "Token missing. Could not authenticate client."
                            " If you have token credentials, see modal.com/docs/reference/modal.config for setup help."
                            " If you are a new user, register an account at modal.com, then run `modal token new`."
                        )
                        raise AuthError(creds_missing_msg)
                    else:
                        raise
                cls._client_from_env = client
                return client

    @classmethod
    async def from_credentials(cls, token_id: str, token_secret: str) -> "_Client":
        """mdmd:hidden
        Constructor based on token credentials; useful for managing Modal on behalf of third-party users.
        """
        server_url = config["server_url"]
        client_type = api_pb2.CLIENT_TYPE_CLIENT
        credentials = (token_id, token_secret)
        client = _Client(server_url, client_type, credentials)
        await client._open()
        try:
            await client._init()
        except BaseException:
            await client._close()
            raise
        async_utils.on_shutdown(client._close())
        return client

    @classmethod
    async def verify(cls, server_url: str, credentials: Tuple[str, str]) -> None:
        """mdmd:hidden
        Check whether can the client can connect to this server with these credentials; raise if not.
        """
        async with cls(server_url, api_pb2.CLIENT_TYPE_CLIENT, credentials):
            pass  # Will call ClientHello RPC and possibly raise AuthError or ConnectionError

    @classmethod
    def set_env_client(cls, client: Optional["_Client"]):
        """mdmd:hidden"""
        # Just used from tests.
        cls._client_from_env = client


Client = synchronize_api(_Client)
