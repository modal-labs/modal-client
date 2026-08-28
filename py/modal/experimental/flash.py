# Copyright Modal Labs 2025
import asyncio
import os
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from modal._clustered_functions import get_cluster_info
from modal._partial_function import _PartialFunctionFlags
from modal.cls import _Cls
from modal_proto import api_pb2

from .._runtime.task_lifecycle_manager import UserException
from .._server import validate_http_server_config
from .._tunnel import _forward as _forward_tunnel
from .._utils.async_utils import synchronize_api, synchronizer
from ..client import _Client
from ..config import logger
from ..exception import InvalidError

_FLASH_UPSTREAM_HEADER = "modal-flash-upstream"
_MAX_FAILURES = 10


class _FlashManager:
    def __init__(
        self,
        client: _Client,
        port: int,
        process: subprocess.Popen | None = None,  # to be deprecated
        health_check_url: str | None = None,
        startup_timeout: int = 30,
        exit_grace_period: int = 0,
        h2_enabled: bool = False,
        is_server: bool = False,
    ):
        self.client = client
        self.port = port
        self.process = process
        # Health check is not currently being used
        self.health_check_url = health_check_url
        self.startup_timeout = startup_timeout
        self.exit_grace_period = exit_grace_period
        self.tunnel_manager = _forward_tunnel(port, h2_enabled=h2_enabled, client=client)
        self.is_server = is_server
        self.stopped = False
        self.num_heartbeat_failures = 0
        self.task_id = os.environ["MODAL_TASK_ID"]
        self.heartbeat_task = None

    async def is_port_connection_healthy(
        self, process: subprocess.Popen | None, timeout: float = 0.5
    ) -> tuple[bool, Exception | None]:
        start_time = time.monotonic()

        def check_process_is_running() -> Exception | None:
            if process is not None and process.poll() is not None:
                return Exception(f"Process {process.pid} exited with code {process.returncode}")
            return None

        while time.monotonic() - start_time < timeout:
            try:
                if error := check_process_is_running():
                    return False, error
                _, writer = await asyncio.wait_for(asyncio.open_connection("localhost", self.port), timeout=0.5)
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
                return True, None
            except asyncio.CancelledError:
                raise
            except (OSError, asyncio.TimeoutError):
                await asyncio.sleep(0.1)

        return False, Exception(f"Waited too long for port {self.port} to accept connections")

    async def _start(self):
        self.tunnel = await self.tunnel_manager.__aenter__()
        parsed_url = urlparse(self.tunnel.url)
        host = parsed_url.hostname
        assert host is not None, f"Tunnel URL has no host: {self.tunnel.url}"
        port = parsed_url.port or 443

        if self.is_server:
            await self._start_server_tunnel()
            return
        await self._start_flash_registration(host, port)

    async def _start_server_tunnel(self) -> None:
        # Worker-side HTTP relay owns Flash registration and drain for server tasks.
        logger.warning(f"[Modal Flash] Server tunnel opened at {self.tunnel.url}.")

    async def _start_flash_registration(self, host: str, port: int) -> None:
        try:
            await self._wait_for_port_success(host, port)
        except (Exception, KeyboardInterrupt, asyncio.CancelledError):
            await self._deregister()
            await self.tunnel_manager.__aexit__(*sys.exc_info())
            raise
        self.heartbeat_task = asyncio.create_task(self._run_heartbeat(host, port))
        self.drain_task = asyncio.create_task(self._drain_container())

    async def _deregister(self):
        await asyncio.shield(
            self.client.stub.FlashContainerDeregister(
                api_pb2.FlashContainerDeregisterRequest(),
                timeout=2,
                retry=None,
            )
        )

    async def _drain_container(self):
        """
        Background task that checks if we've encountered too many failures and drains the container if so.
        """
        while True:
            try:
                # Check if the container should be drained (e.g., too many failures)
                if self.num_heartbeat_failures > _MAX_FAILURES:
                    logger.warning(
                        f"[Modal Flash] Draining task {self.task_id} on {self.tunnel.url} due to too many failures."
                    )
                    await self.stop()
                    # handle close upon container exit

                    if self.task_id:
                        await self.client.stub.ContainerStop(api_pb2.ContainerStopRequest(task_id=self.task_id))
                    return
            except asyncio.CancelledError:
                logger.warning("[Modal Flash] Shutting down...")
                return
            except Exception as e:
                logger.error(f"[Modal Flash] Error draining container: {e}")
                await asyncio.sleep(1)

            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.warning("[Modal Flash] Shutting down...")
                return

    async def _wait_for_port_success(self, host: str, port: int) -> bool:
        start_time = time.monotonic()
        while time.monotonic() - start_time < self.startup_timeout:
            try:
                port_check_resp, _ = await self.is_port_connection_healthy(process=self.process)
                if port_check_resp:
                    resp = await self.client.stub.FlashContainerRegister(
                        api_pb2.FlashContainerRegisterRequest(
                            priority=10,
                            weight=5,
                            host=host,
                            port=port,
                        ),
                        timeout=10,
                        retry=None,
                    )
                    logger.warning(f"Listening at {resp.url} over {self.tunnel.url} for task_id {self.task_id}")
                    return True
            except asyncio.CancelledError:
                logger.warning("Healthcheck cancelled while waiting for port to accept connections. Shutting down...")
                raise
            except Exception as e:
                logger.error(f"Error waiting for port to accept connections: {e}")
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.warning("Healthcheck cancelled while waiting for port to accept connections. Shutting down...")
                raise
        raise TimeoutError("Timed out while waiting for port to accept connections. Shutting down...")

    async def _run_heartbeat(self, host: str, port: int):
        while True:
            try:
                port_check_resp, port_check_error = await self.is_port_connection_healthy(process=self.process)
                if port_check_resp:
                    resp = await self.client.stub.FlashContainerRegister(
                        api_pb2.FlashContainerRegisterRequest(
                            priority=10,
                            weight=5,
                            host=host,
                            port=port,
                        ),
                        timeout=10,
                        retry=None,
                    )
                    self.num_heartbeat_failures = 0
                else:
                    logger.error(
                        f"[Modal Flash] Deregistering container {self.task_id} on {self.tunnel.url} "
                        f"due to error: {port_check_error}, num_heartbeat_failures: {self.num_heartbeat_failures}"
                    )
                    self.num_heartbeat_failures += 1
                    await self._deregister()
            except asyncio.CancelledError:
                logger.warning("[Modal Flash] Shutting down...")
                await self._deregister()
                break
            except Exception as e:
                logger.error(f"[Modal Flash] Heartbeat failed: {e}")
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                await self._deregister()
                break

    def get_container_url(self):
        # WARNING: Try not to use this method; we aren't sure if we will keep it.
        return self.tunnel.url

    async def stop(self):
        try:
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                try:
                    # NOTE(gongy): We skip calling TunnelStop to avoid interrupting in-flight requests.
                    # It is up to the user to wait after calling .stop() to drain in-flight requests.
                    await asyncio.wait_for(self.heartbeat_task, timeout=5)
                    logger.warning(f"[Modal Flash] Stopping heartbeat task on {self.tunnel.url}.")
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning("[Modal Flash] Heartbeat task did not stop within 5s.")
        except Exception as e:
            logger.error(f"[Modal Flash] Error stopping: {e}")
        self.stopped = True

    async def close(self):
        if not self.stopped:
            await self.stop()

        # Server tasks drain via the worker-side HTTP relay, so skip the
        # Python-side sleep here to avoid double-counting the grace period.
        if not self.is_server:
            await asyncio.sleep(self.exit_grace_period)

        logger.warning(f"[Modal Flash] Closing tunnel on {self.tunnel.url}.")
        await self.tunnel_manager.__aexit__(*sys.exc_info())


FlashManager = synchronize_api(_FlashManager, target_module=__name__)


@synchronizer.create_blocking
async def flash_forward(
    port: int,
    process: subprocess.Popen | None = None,  # to be deprecated
    health_check_url: str | None = None,
    startup_timeout: int = 30,
    exit_grace_period: int = 0,
    h2_enabled: bool = False,
    is_server: bool = False,
) -> _FlashManager:
    """
    Forward a port to the Modal Flash service, exposing that port as a stable endpoint.
    This is a highly experimental method that can break or be removed at any time without warning.
    Do not use this method unless explicitly instructed to do so by Modal support.
    """
    client = await _Client.from_env()

    manager = _FlashManager(
        client,
        port,
        process=process,
        health_check_url=health_check_url,
        startup_timeout=startup_timeout,
        exit_grace_period=exit_grace_period,
        h2_enabled=h2_enabled,
        is_server=is_server,
    )
    await manager._start()
    return manager


@synchronizer.create_blocking
async def flash_get_containers(app_name: str, cls_name: str) -> list[Any]:
    """
    Return a list of flash containers for a deployed Flash service.

    Each entry exposes `task_id`, `host`, and `port` attributes.

    This is a highly experimental method that can break or be removed at any time without warning.
    Do not use this method unless explicitly instructed to do so by Modal support.
    """
    client = await _Client.from_env()
    fn = _Cls.from_name(app_name, cls_name)._get_class_service_function()
    await fn.hydrate(client=client)
    req = api_pb2.FlashContainerListRequest(function_id=fn.object_id)
    resp = await client.stub.FlashContainerList(req)
    return list(resp.containers)


def _http_server(
    port: int | None = None,
    *,
    proxy_regions: list[str] = [],  # The regions to proxy the HTTP server to.
    startup_timeout: int = 30,  # Maximum number of seconds to wait for the HTTP server to start.
    exit_grace_period: int | None = None,  # The time to wait for the HTTP server to exit gracefully.
    h2_enabled: bool = False,  # Whether to enable HTTP/2 support.
):
    """Decorator for Flash-enabled HTTP servers on Modal classes.

    Args:
        port: The local port to forward to the HTTP server.
        proxy_regions: The regions to proxy the HTTP server to.
        startup_timeout: The maximum time to wait for the HTTP server to start.
        exit_grace_period: The time to wait for the HTTP server to exit gracefully.

    """
    if port is None:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@modal.http_server()`."
        )
    validate_http_server_config(port, proxy_regions, startup_timeout, exit_grace_period, is_server=False)

    from modal._partial_function import _PartialFunction, _PartialFunctionParams

    params = _PartialFunctionParams(
        http_config=api_pb2.HTTPConfig(
            port=port,
            proxy_regions=proxy_regions,
            startup_timeout=startup_timeout or 0,
            exit_grace_period=exit_grace_period or 0,
            h2_enabled=h2_enabled,
        )
    )

    def wrapper(obj: Callable[..., Any] | _PartialFunction) -> _PartialFunction:
        flags = _PartialFunctionFlags.HTTP_WEB_INTERFACE

        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags, params)
        else:
            pf = _PartialFunction(obj, flags, params)
        pf.validate_obj_compatibility("`http_server`")
        return pf

    return wrapper


http_server = synchronize_api(_http_server, target_module=__name__)


class _FlashContainerEntry:
    """
    A class that manages the lifecycle of Flash manager for Flash containers.

    It is intentional that stop() runs before exit handlers and close().
    This ensures the container is deregistered first, preventing new requests from being routed to it
    while exit handlers execute and the exit grace period elapses, before finally closing the tunnel.
    """

    flash_manager: FlashManager | None  # type: ignore

    def __init__(self, http_config: api_pb2.HTTPConfig, is_server: bool = False):
        self.http_config: api_pb2.HTTPConfig = http_config
        self.flash_manager = None
        self.is_server = is_server

    def enter(self):
        if self.http_config != api_pb2.HTTPConfig():
            try:
                rank = get_cluster_info().rank
                if rank != 0:
                    return
            except InvalidError:
                pass
            try:
                self.flash_manager = flash_forward(
                    self.http_config.port,
                    startup_timeout=self.http_config.startup_timeout,
                    exit_grace_period=self.http_config.exit_grace_period,
                    h2_enabled=self.http_config.h2_enabled,
                    is_server=self.is_server,
                )
            except Exception as e:
                logger.warning(f"[Modal Flash] Startup failed: {e}")
                raise UserException()

    def stop(self):
        if self.flash_manager:
            self.flash_manager.stop()

    def close(self):
        if self.flash_manager:
            self.flash_manager.close()
