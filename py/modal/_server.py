# Copyright Modal Labs 2025
import inspect
import json
import typing
from datetime import datetime, timedelta, timezone

from typing_extensions import Self

from modal_proto import api_pb2

from ._functions import _Function
from ._load_context import LoadContext
from ._logs_manager import _ServerLogsManager
from ._object import live_method
from ._partial_function import (
    _find_partial_methods_for_user_cls,
    _PartialFunction,
    _PartialFunctionFlags,
)
from ._supports_logs import _LogQueryData
from ._utils.async_utils import retry, synchronize_api
from ._utils.http_utils import ClientSessionRegistry
from .client import _Client
from .cls import is_parameter
from .config import logger
from .exception import ExecutionError, InvalidError, ServiceError
from .types import ServerAutoscalerSettings, ServerContainerInfo, ServerInfo, ServerSessionCredentials, ServerStats

if typing.TYPE_CHECKING:
    import modal.app


def validate_http_server_config(
    port: int,
    proxy_regions: list[str],  # The regions to proxy the HTTP server to.
    startup_timeout: int,  # Maximum number of seconds to wait for the HTTP server to start.
    exit_grace_period: int | None,  # The time to wait for the HTTP server to exit gracefully.
    is_server: bool = False,  # Whether this validates a `server` config.
):
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise InvalidError("Port must be a positive integer between 1 and 65535.")
    if startup_timeout <= 0:
        raise InvalidError("The `startup_timeout` argument must be positive.")
    if exit_grace_period is not None and exit_grace_period < 0:
        raise InvalidError("The `exit_grace_period` argument must be non-negative.")
    if is_server:
        if exit_grace_period is not None and exit_grace_period > 3600:
            raise InvalidError("The `exit_grace_period` argument must not exceed 3600 seconds (1 hour).")
    elif exit_grace_period is not None and exit_grace_period > 25:
        raise InvalidError("The `exit_grace_period` argument must not exceed 25 seconds.")

    if not proxy_regions or not proxy_regions[0]:
        if is_server:
            raise InvalidError("The `routing_region` argument must be passed.")
        raise InvalidError("The `proxy_regions` argument must be non-empty.")


class _Server:
    """Server runs an HTTP server started in an `@modal.enter` method.

    See the [guide](https://modal.com/docs/guide/servers) for more information.

    Generally, you will not construct a Server directly.
    Instead, use the [`@app.server()`](https://modal.com/docs/sdk/py/latest/App#server) decorator.

    ```python notest
    @app.server(port=8080, routing_region="us-east")
    class MyServer:
        @modal.enter()
        def start_server(self):
            self.process = subprocess.Popen(["python3", "-m", "http.server", "8080"])
    ```
    """

    _user_cls: type | None = None  # None if remote
    _service_function: _Function
    _app: "modal.app._App | None" = None  # None if remote
    _is_sessioned: bool | None = None  # None until known (via local set or remote hydration)

    def _get_user_cls(self) -> type:
        assert self._user_cls is not None
        return self._user_cls

    def _get_app(self) -> "modal.app._App":
        assert self._app
        return self._app

    def _get_service_function(self) -> _Function:
        return self._service_function

    @property
    def object_id(self) -> str:
        """Modal's internal ID for this Server instance."""
        return self._service_function.object_id

    async def _get_log_query_data(self) -> _LogQueryData:
        return await self._get_service_function()._get_log_query_data()

    @property
    def logs(self) -> _ServerLogsManager:
        """Access logs for a `Server`.

        Use [`fetch()`](#logsfetch)
        to read logs from a UTC time range, [`tail()`](#logstail)
        to read the most recent logs, and [`stream()`](#logsstream)
        to follow new logs as they arrive.

        See also:
            - [`modal app logs`](https://modal.com/docs/cli/latest/app#modal-app-logs):
            CLI access to logs for an App.
        """
        return _ServerLogsManager(self)

    @property
    def sessions(self) -> "_ServerSessionsManager":
        """Start and terminate sessions on a Server decorated with `@modal.sessioned()`."""
        return _ServerSessionsManager(self)

    async def info(self, *, refresh: bool = False) -> ServerInfo:
        """Get an overview of a Server's resource requests, associated mounts, http config, etc.

        This method performs a network request to populate this information if the Server handle is
        a remote lookup whose information has not yet been fetched (e.g. from `Server.from_name(...)`),
        or if `refresh=True`.

        Args:
            refresh: Always perform a network request. Pass `refresh=True` to ensure that this method
                returns the most up to date information.

        Returns:
            This returns a [`modal.types.ServerInfo`](https://modal.com/docs/sdk/py/latest/types#ServerInfo)
            dataclass.
        """

        return ServerInfo._from_function_info(await self._get_service_function().info(refresh=refresh))

    @staticmethod
    def _extract_user_cls(wrapped_user_cls: "type | _PartialFunction") -> type:
        if isinstance(wrapped_user_cls, _PartialFunction):
            assert wrapped_user_cls.user_cls
            return wrapped_user_cls.user_cls
        else:
            return wrapped_user_cls

    # ============ Live Methods ============

    @live_method
    async def get_url(self) -> str | None:
        """The URL for making requests to this Server."""
        urls = await self._get_service_function()._experimental_get_flash_urls()
        # Return the first URL if it exists. Servers should only have one URL
        # since they only have one region.
        url = urls[0] if urls else None
        return url

    @live_method
    async def _experimental_list_containers(self) -> list[ServerContainerInfo]:
        """List the containers currently registered to serve requests for this Server.

        This interface is experimental and may change or be removed without warning.
        """
        service_function = self._get_service_function()
        response = await service_function.client.stub.FlashContainerList(
            api_pb2.FlashContainerListRequest(function_id=service_function.object_id)
        )
        return [
            ServerContainerInfo(
                container_id=container.task_id,
                host=container.host,
                port=container.port,
            )
            for container in response.containers
        ]

    @live_method
    async def update_autoscaler(
        self,
        *,
        target_concurrency: float | None = None,
        min_containers: int | None = None,
        max_containers: int | None = None,
        buffer_containers: int | None = None,
        scaleup_window: int | None = None,
        scaledown_window: int | None = None,
    ) -> ServerAutoscalerSettings:
        """Override the current autoscaler behavior for this Server.

        Unspecified parameters will retain their current value, i.e. either the static value
        from the `@app.server()` decorator, or an override value from a previous call to this method.

        Subsequent deployments of the App containing this Server will reset the autoscaler back to
        its static configuration.

        Args:
            target_concurrency:
                Target number of concurrent requests per container. May be fractional, e.g. 1.5 to
                target three concurrent requests per two containers.
            min_containers: Minimum number of containers to keep running regardless of demand.
            max_containers: Limit on the number of containers that can be concurrently running.
            buffer_containers: Extra containers to scale up beyond current demand.
            scaleup_window: Seconds of sustained demand required before scaling up new containers.
            scaledown_window: Maximum duration (in seconds) idle containers wait before scaling down.

        Returns:
            A `ServerAutoscalerSettings` dataclass which contains the current autoscaler settings of
            this Server after the call.

        Examples:
            ```python notest
            server = modal.Server.from_name("my-app", "Server")

            # Always have at least 2 containers running, with an extra buffer of 2 containers
            server.update_autoscaler(min_containers=2, buffer_containers=1)

            # Limit this Server to avoid spinning up more than 5 containers
            server.update_autoscaler(max_containers=5)

            # Require 30 seconds of sustained demand before scaling up
            server.update_autoscaler(scaleup_window=30)

            # Adjust Server autoscaling to target 20 concurrent requests per replica
            server.update_autoscaler(target_concurrency=20)

            # Target three concurrent requests for every two containers
            server.update_autoscaler(target_concurrency=1.5)

            # Disable the Server autoscaling by setting target_concurrency to 0
            server.update_autoscaler(target_concurrency=0)
            ```

        """
        return await self._get_service_function()._update_autoscaler_server(
            min_containers=min_containers,
            max_containers=max_containers,
            scaleup_window=scaleup_window,
            scaledown_window=scaledown_window,
            buffer_containers=buffer_containers,
            target_concurrency=target_concurrency,
        )

    # ============ Hydration ============
    async def hydrate(self, client: _Client | None = None) -> "_Server":
        """Synchronize the local object with its identity on the Modal server.

        It is rarely necessary to call this method explicitly, as most operations will
        lazily hydrate when needed. The main use case is when you need to access object
        metadata, such as its ID.

        """
        # This is required since we want to support @livemethod() decorated methods
        service_function = self._get_service_function()
        await service_function.hydrate(client)
        if service_function._function_info is not None:
            self._is_sessioned = service_function._function_info._sessioned
        return self

    @classmethod
    def _new_from_function(cls, object_id: str, client: _Client, metadata: api_pb2.FunctionHandleMetadata) -> Self:
        """mdmd:hidden

        Callers which already have handle metadata for a Server service function can use this method to
        create a hydrated handle without having to do an unnecessary RPC.
        """

        obj = cls()
        obj._service_function = _Function._new_hydrated(object_id, client, metadata)
        return obj

    # ============ Construction ============
    @staticmethod
    def _from_local(
        wrapped_user_cls: "type | _PartialFunction",
        app: "modal.app._App",
        service_function: _Function,
        is_sessioned: bool = False,
    ) -> "_Server":
        """Create a Server from a local class definition."""

        # Note: Validation should be done by the caller (app.server()) BEFORE creating the Server.
        # Extract the underlying class if wrapped in a _PartialFunction (e.g., from @modal.clustered())
        user_cls = _Server._extract_user_cls(wrapped_user_cls)

        server = _Server()
        server._app = app
        server._user_cls = user_cls
        server._service_function = service_function
        server._is_sessioned = is_sessioned
        return server

    @classmethod
    def from_name(
        cls: type["_Server"],
        app_name: str,
        name: str,
        *,
        environment_name: str | None = None,
        client: _Client | None = None,
    ) -> "_Server":
        """Reference a Server from a deployed App by its name.

        This is a lazy method that defers hydrating the local
        object with metadata from Modal servers until the first
        time it is actually used.

        Args:
            app_name: Name of the App containing the Server.
            name: Name of the Server within the App.
            environment_name: Name of the Environment where the App is deployed.
            client: Modal client instance for this session.

        ```python notest
        server = modal.Server.from_name("other-app", "Server")
        ```
        """

        load_context_overrides = LoadContext(client=client, environment_name=environment_name)

        server = _Server()
        server._service_function = _Function._from_name(
            app_name,
            name,
            load_context_overrides=load_context_overrides,
        )
        return server

    @classmethod
    def from_id(
        cls: type["_Server"],
        server_id: str,
        *,
        client: _Client | None = None,
    ):
        """Reference a Server from a deployed or running App by its ID.

        This is a lazy method that defers hydrating the local
        object with metadata from Modal servers until the first
        time it is actually used.

        Args:
            server_id: The ID of the server.
            client: Modal client instance for this session.

        Examples:
            ```python notest
            server = modal.Server.from_id("fu-456")
            ```
        """
        load_context_overrides = LoadContext(client=client)

        server = _Server()
        server._service_function = _Function._from_id(
            server_id,
            load_context_overrides=load_context_overrides,
            called_from="Server",
        )
        return server

    def _is_local(self) -> bool:
        """Returns True if this Server has local source code available."""
        return self._user_cls is not None

    # ============ Validation ============

    @staticmethod
    def _validate_wrapped_user_cls_decorators(
        wrapped_user_cls: "type | _PartialFunction", enable_memory_snapshot: bool
    ):
        user_cls = _Server._extract_user_cls(wrapped_user_cls)

        if not inspect.isclass(user_cls):
            raise TypeError("The @app.server() decorator must be used on a class.")

        # Check for modal.parameter() - not allowed on server classes
        params = {k: v for k, v in user_cls.__dict__.items() if is_parameter(v)}
        if params:
            raise InvalidError(
                f"Server class {user_cls.__name__} cannot use modal.parameter(). "
                "Servers do not support parameterization."
            )

        if not _find_partial_methods_for_user_cls(
            user_cls, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT
        ) and not _find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.ENTER_POST_SNAPSHOT):
            raise InvalidError("Server class must have an @modal.enter() to setup the server.")

        # Check for disallowed decorators
        # @modal.method() not allowed
        if _find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.CALLABLE_INTERFACE).values():
            raise InvalidError(
                f"Server class {user_cls.__name__} cannot have `@modal.method()` decorated functions. "
                "Servers only expose HTTP endpoints."
            )
        # @enter with snap=True without enable_memory_snapshot
        if (
            _find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT)
            and not enable_memory_snapshot
        ):
            raise InvalidError(
                "Server must have `enable_memory_snapshot=True` to use `snap=True` on `@modal.enter` methods."
            )

        if isinstance(wrapped_user_cls, _PartialFunction):
            # @modal.concurrent not allowed on server classes
            if wrapped_user_cls.flags & _PartialFunctionFlags.CONCURRENT:
                raise InvalidError(
                    f"Server class {user_cls.__name__} cannot be decorated with `@modal.concurrent()`. "
                    "Please use `target_concurrency` param instead."
                )
            # @modal.http_server not allowed on server classes
            if wrapped_user_cls.flags & _PartialFunctionFlags.HTTP_WEB_INTERFACE:
                raise InvalidError(
                    f"Server class {user_cls.__name__} cannot have @modal.experimental.http_server() decorator. "
                    "Servers already expose HTTP endpoints."
                )
            # @modal.web_server not allowed on server classes
            if wrapped_user_cls.flags & _PartialFunctionFlags.WEB_INTERFACE:
                raise InvalidError(
                    f"Server class {user_cls.__name__} cannot be decorated with `@modal.web_server()`. "
                    "Servers already expose HTTP endpoints."
                )

    @staticmethod
    def _validate_construction_mechanism(wrapped_user_cls: "type | _PartialFunction"):
        """Validate that the server class doesn't have a custom constructor."""
        # Extract the underlying class if wrapped in a _PartialFunction (e.g., from @modal.clustered())
        user_cls = _Server._extract_user_cls(wrapped_user_cls)

        if user_cls.__init__ != object.__init__:  # type: ignore
            raise InvalidError(
                f"Server class {user_cls.__name__} cannot have a custom __init__ method. "
                "Use @modal.enter() for initialization logic instead."
            )

    @live_method
    async def stats(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        container: str | None = None,
    ) -> ServerStats:
        """Return statistics for a modal Server.

        The default time range is the most recent hour. The maximum time range is 7 days.

        Args:
            since: The beginning of the time range, inclusive. If omitted, this defaults to an hour before `until`.
               Values without a timezone are interpeted as local time.
            until: The end of the time range, exclusive. If omitted, this defaults to current time.
                Values without a timezone are interpeted as local time.
            container: If passed in, the stats are computed for only this container. Default None.

        Returns:
            A `ServerStats` object
        """
        until = until or datetime.now(timezone.utc)
        if until.tzinfo is None:
            until = until.astimezone()
        until = until.astimezone(timezone.utc)

        since = since or until - timedelta(hours=1)
        if since.tzinfo is None:
            since = since.astimezone()
        since = since.astimezone(timezone.utc)
        if since >= until:
            raise InvalidError("`since` must be before `until`.")

        request = api_pb2.ServerGetTimeRangeStatsRequest(function_id=self.object_id)
        request.since.FromDatetime(since)
        request.until.FromDatetime(until)
        if container:
            request.container_id = container
        stats = await self._get_service_function().client.stub.ServerGetTimeRangeStats(request)
        return ServerStats._from_proto(stats)


@retry(n_attempts=5, base_delay=0.5, attempt_timeout=65, total_timeout=200)
async def _post_session_control(url: str, headers: dict[str, str]) -> tuple[int, str, str]:
    """POST to a session control endpoint, retrying connection errors and 5xx. Returns (status, reason, body)."""
    async with ClientSessionRegistry.get_session().post(url, headers=headers) as resp:
        body = await resp.text()
        if resp.status >= 400:
            logger.debug(f"Session control request to {url} failed with status {resp.status}")
        if resp.status >= 500:
            raise ServiceError(f"status {resp.status} {resp.reason}")
        return resp.status, resp.reason or "", body


class _ServerSessionsManager:
    """mdmd:namespace"""

    def __init__(self, server: "_Server"):
        """mdmd:hidden"""
        self._server = server

    def _validate(self) -> None:
        if self._server._is_sessioned is False:
            raise InvalidError("`sessions` requires `@modal.sessioned()` on the Server.")

    async def start(self, idle_timeout: int = 600) -> ServerSessionCredentials:
        """Start a session and return its ID and token.

        Requests to the server URL that carry the returned token are routed to the same container until the session
        has had no connections for `idle_timeout` seconds or is terminated. A container won't be scaled down for as long
        as it holds a live session.

        Args:
            idle_timeout: Seconds without an in-flight request before the session ends.

        Examples:

            ```python notest
            server = modal.Server.from_name("my-app", "MyServer")
            server_url = server.get_url()
            session = server.sessions.start(idle_timeout=600)
            headers = {"Modal-Authorization": f"Bearer {session.token}"}

            requests.get(server_url, headers=headers).raise_for_status()

            server.sessions.terminate(session.token)
            ```
        """
        if not isinstance(idle_timeout, int) or idle_timeout <= 0:
            raise InvalidError("`idle_timeout` must be a positive integer.")

        fn = self._server._get_service_function()
        url = await self._server.get_url()
        self._validate()
        assert url is not None, "Server has no URL."

        headers = {
            "Modal-Authorization": f"Bearer {await fn._get_flash_auth_token()}",
            "x-modal-server-session-idle-timeout": str(idle_timeout),
        }

        try:
            status, reason, body = await _post_session_control(f"{url}/_modal/sessions/start", headers)
        except ServiceError as exc:
            raise ExecutionError(f"Failed to start session: {exc}") from None
        if status >= 400:
            raise ExecutionError(f"Failed to start session: status {status} {reason}")
        try:
            data = json.loads(body)
            return ServerSessionCredentials(session_id=data["session_id"], token=data["token"])
        except (ValueError, KeyError, TypeError) as exc:
            logger.debug(f"Malformed session start response: {exc!r}")
            raise ExecutionError("Failed to start session: unexpected response from server") from None

    async def terminate(self, token: str) -> None:
        """Terminate a session. New requests to it will be rejected. Container will continue serving other sessions.

        Args:
            token: The `token` of the `ServerSessionCredentials` to terminate.

        Examples:

            ```python notest
            server = modal.Server.from_name("my-app", "MyServer")
            session = server.sessions.start()

            server.sessions.terminate(session.token)
            ```
        """
        fn = self._server._get_service_function()
        url = await self._server.get_url()
        self._validate()
        assert url is not None, "Server has no URL."

        headers = {
            "Modal-Authorization": f"Bearer {await fn._get_flash_auth_token()}",
            "x-modal-server-session-token": token,
        }
        try:
            status, reason, _body = await _post_session_control(f"{url}/_modal/sessions/terminate", headers)
        except ServiceError as exc:
            raise ExecutionError(f"Failed to terminate session: {exc}") from None
        if status == 404:
            # Treat 404 as success, assuming session was already terminated
            return
        if status >= 400:
            raise ExecutionError(f"Failed to terminate session: status {status} {reason}")


ServerSessionsManager = synchronize_api(_ServerSessionsManager, target_module=__name__)
