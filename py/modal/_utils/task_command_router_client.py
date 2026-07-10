# Copyright Modal Labs 2025
import asyncio
import base64
import json
import socket
import ssl
import time
import typing
import urllib.parse
import weakref
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from typing import TypeVar

import grpclib.client
from grpclib import GRPCError, Status
from grpclib.exceptions import StreamTerminatedError

from modal.config import logger
from modal.exception import ExecTimeoutError, TimeoutError as ModalTimeoutError
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2
from modal_proto.task_command_router_grpc import TaskCommandRouterStub

from .._grpc_client import grpc_error_converter
from .._utils.grpc_utils import ModalChannel, create_channel_config
from .async_utils import aclosing, retry
from .grpc_utils import RETRYABLE_GRPC_STATUS_CODES


@retry(n_attempts=34, base_delay=1, max_delay=10, attempt_timeout=10, total_timeout=310)
async def _connect_channel(channel: grpclib.client.Channel):
    """Connect to the command router channel.

    Uses a longer retry budget than grpc_utils.create_channel_with_fallbacks. In rare cases the
    sandbox may take a long time to start on the worker after scheduling.

    Retries with exponential backoff (1, 2, 4, 8, 10, 10, ...) capped at 10s per delay.
    Total sleep between attempts: 1 + 2 + 4 + 8 + 10*29 = 305s (~5 min).
    """
    await channel.__connect__()


def _b64url_decode(data: str) -> bytes:
    """Decode a base64url string with missing padding tolerated."""
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _parse_jwt_expiration(jwt_token: str) -> float | None:
    """Parse exp from a JWT without verification. Returns UNIX time seconds or None.

    This is best-effort; if parsing fails or claim missing, returns None.
    """
    try:
        parts = jwt_token.split(".")
        if len(parts) != 3:
            return None
        payload_b = _b64url_decode(parts[1])
        payload = json.loads(payload_b)
        exp = payload.get("exp")
        if isinstance(exp, (int, float)):
            return float(exp)
    except Exception:
        # Avoid raising on malformed tokens; fall back to server-driven refresh logic.
        logger.warning("Failed to parse JWT expiration")
        return None
    return None


async def call_with_retries_on_transient_errors(
    func,
    *,
    base_delay_secs: float = 0.01,
    delay_factor: float = 2,
    max_retries: int | None = 10,
    exclude_status_codes: list[Status] | None = None,
    timeout_deadline: float | None = None,
):
    """Call func() with transient error retries and exponential backoff.

    Authentication retries are expected to be handled by the caller.

    Args:
        exclude_status_codes: gRPC status codes to exclude from retry logic even if
            they are in RETRYABLE_GRPC_STATUS_CODES. Use this to let certain errors
            (e.g. DEADLINE_EXCEEDED) propagate immediately rather than being retried.
        timeout_deadline: Optional monotonic deadline (`time.monotonic()` value).
            When set, retries are not attempted once the deadline is reached and
            the backoff sleep is clamped so we don't sleep past it. It's up to
            the caller to decide whether to further translate the surfaced
            exception (e.g., into a TimeoutError) based on the deadline check.
            The caller is also responsible for propagating the remaining budget
            into `func()` (typically as the per-call gRPC timeout).
    """
    delay_secs = base_delay_secs
    num_retries = 0
    exclude_status_codes = exclude_status_codes or []

    def is_retryable_status(status: Status) -> bool:
        return status in RETRYABLE_GRPC_STATUS_CODES and status not in exclude_status_codes

    def can_retry() -> bool:
        if max_retries is not None and num_retries >= max_retries:
            return False
        if timeout_deadline is not None and time.monotonic() >= timeout_deadline:
            return False
        return True

    async def sleep_and_advance(e: Exception):
        nonlocal delay_secs, num_retries
        # Clamp the backoff sleep to the remaining deadline so we don't sleep
        # past it just to fail on the next iteration's deadline check.
        sleep_for = delay_secs
        if timeout_deadline is not None:
            sleep_for = min(sleep_for, max(0.0, timeout_deadline - time.monotonic()))
        logger.debug(f"Retrying RPC with delay {sleep_for}s due to error: {e}")
        await asyncio.sleep(sleep_for)
        delay_secs *= delay_factor
        num_retries += 1

    while True:
        try:
            return await func()
        except GRPCError as e:
            if not is_retryable_status(e.status) or not can_retry():
                raise
            await sleep_and_advance(e)
        except AttributeError as e:
            # StreamTerminatedError are not properly raised in grpclib<=0.4.7
            # fixed in https://github.com/vmagamedov/grpclib/issues/185
            # TODO: update to newer version (>=0.4.8) once stable
            if "_write_appdata" not in str(e) or not can_retry():
                raise
            await sleep_and_advance(e)
        except StreamTerminatedError as e:
            if not can_retry():
                raise
            await sleep_and_advance(e)
        except (asyncio.TimeoutError, OSError) as e:
            if not can_retry():
                # Client-side timeout / network OSError surfaces as a generic
                # ConnectionError once we stop retrying. Callers that pass
                # `timeout_deadline` can further translate this based on
                # whether the deadline has elapsed.
                raise ConnectionError(str(e))
            await sleep_and_advance(e)


_StdioReq = TypeVar("_StdioReq")
_StdioResp = TypeVar("_StdioResp", sr_pb2.TaskExecStdioReadResponse, sr_pb2.SandboxStdioReadV2Response)


async def fetch_command_router_access(server_client, task_id: str) -> api_pb2.TaskGetCommandRouterAccessResponse:
    """Fetch direct command router access info from Modal server."""
    return await server_client.stub.TaskGetCommandRouterAccess(
        api_pb2.TaskGetCommandRouterAccessRequest(task_id=task_id),
    )


async def fetch_command_router_access_v2(
    server_client, sandbox_id: str
) -> api_pb2.SandboxGetCommandRouterAccessResponse:
    """Fetch direct command router access info from Modal server for a V2 sandbox."""
    assert server_client._auth_token_manager
    auth_token = await server_client._auth_token_manager.get_token()
    return await server_client.stub.SandboxGetCommandRouterAccess(
        api_pb2.SandboxGetCommandRouterAccessRequest(sandbox_id=sandbox_id),
        metadata=[("x-modal-auth-token", auth_token)],
    )


def _finalize_channel(loop, channel):
    if not loop.is_closed():
        # only run if loop has not shut down
        # call_soon_threadsafe could throw if the loop is torn down after calling
        # is_closed. This is safe to ignore.
        with suppress(Exception):
            loop.call_soon_threadsafe(channel.close)


class TaskCommandRouterClient:
    """
    Client used to talk directly to TaskCommandRouter service on worker hosts.

    A new instance should be created per task.
    """

    @classmethod
    async def _connect(
        cls,
        server_client,
        task_id: str,
        url: str,
        jwt: str,
        *,
        sandbox_id: str | None = None,
    ) -> "TaskCommandRouterClient":
        """Build a connected client from a jwt and url."""
        o = urllib.parse.urlparse(url)
        is_localhost_client = server_client._is_localhost
        if o.scheme == "http":
            # plain http serving should only be used for unit tests
            if not is_localhost_client:
                raise ValueError(f"Task router URL must be https, got: {url}")
            ssl_context = None
        elif o.scheme == "https":
            ssl_context = ssl.create_default_context()
            if is_localhost_client:
                # Allow insecure TLS when the Modal server client points to localhost
                # This is typically triggered from local e2e testing
                logger.warning("Using insecure TLS for task command router because server client points to localhost")
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
        else:
            raise ValueError(f"Task router URL must be https, got: {url}")

        host, _, port_str = o.netloc.partition(":")
        port = int(port_str) if port_str else (443 if o.scheme == "https" else 80)

        channel = ModalChannel(
            host,
            port,
            ssl=ssl_context,
            config=create_channel_config(),
            closed_error_message="Unable to perform operation on a detached sandbox",
        )

        async def send_request(event: grpclib.events.SendRequest) -> None:
            idempotency_key = typing.cast(str | None, event.metadata.get("x-idempotency-key"))
            if idempotency_key is None:
                logger.debug(f"Sending request to {event.method_name}")
            else:
                logger.debug(f"Sending request to {event.method_name} ({idempotency_key[:8]})")

        grpclib.events.listen(channel, grpclib.events.SendRequest, send_request)

        try:
            await _connect_channel(channel)
        except socket.gaierror as exc:
            raise ConnectionError(f"Could not resolve hostname '{host}': {exc}") from exc
        loop = asyncio.get_running_loop()
        jwt_refresh_lock = asyncio.Lock()

        return cls(server_client, task_id, url, jwt, channel, loop, jwt_refresh_lock, sandbox_id=sandbox_id)

    @classmethod
    async def init(
        cls,
        server_client,
        task_id: str,
    ) -> "TaskCommandRouterClient":
        resp = await fetch_command_router_access(server_client, task_id)
        logger.debug(f"Using command router access for task {task_id}")
        return await cls._connect(server_client, task_id, resp.url, resp.jwt)

    @classmethod
    async def init_v2(
        cls,
        server_client,
        sandbox_id: str,
        task_id: str,
    ) -> "TaskCommandRouterClient":
        """Initialize a TaskCommandRouterClient for a V2 sandbox."""
        resp = await fetch_command_router_access_v2(server_client, sandbox_id)
        logger.debug(f"Using command router access for sandbox {sandbox_id}")
        return await cls._connect(server_client, task_id, resp.url, resp.jwt, sandbox_id=sandbox_id)

    def __init__(
        self,
        server_client,
        task_id: str,
        server_url: str,
        jwt: str,
        channel: grpclib.client.Channel,
        loop: asyncio.AbstractEventLoop,
        jwt_refresh_lock: asyncio.Lock,
        *,
        sandbox_id: str | None = None,
        stream_stdio_retry_delay_secs: float = 0.01,
        stream_stdio_retry_delay_factor: float = 2,
        stream_stdio_max_retries: int = 10,
    ) -> None:
        """Callers should not use this directly. Use TaskCommandRouterClient.init() instead."""
        # Record the loop this instance is bound to so __del__ can safely schedule cleanup
        # even if finalization happens from a different thread (e.g. via synchronicity).
        self._loop = loop

        # Attach bearer token on all requests to the worker-side router service.
        self._server_client = server_client
        self._task_id = task_id
        self._sandbox_id = sandbox_id
        self._server_url = server_url
        self._jwt = jwt
        self._channel = channel
        # Retry configuration for stdio streaming
        self.stream_stdio_retry_delay_secs = stream_stdio_retry_delay_secs
        self.stream_stdio_retry_delay_factor = stream_stdio_retry_delay_factor
        self.stream_stdio_max_retries = stream_stdio_max_retries

        # JWT refresh coordination
        self._jwt_exp: float | None = _parse_jwt_expiration(jwt)
        # This is passed in as an argument to ensure it's created from within the correct event loop.
        self._jwt_refresh_lock = jwt_refresh_lock

        self._closed = False

        self._channel_finalizer = weakref.finalize(
            self,
            _finalize_channel,
            loop,
            channel,
        )

        self._stub = TaskCommandRouterStub(self._channel)

    @property
    def _is_v2_sandbox(self) -> bool:
        return self._sandbox_id is not None

    def _get_metadata(self):
        return {"authorization": f"Bearer {self._jwt}"}

    async def close(self) -> None:
        """Close the client."""
        if self._closed:
            return

        self._closed = True
        self._channel.close()
        if self._channel_finalizer.alive:
            # skip the finalizer if we've closed the channel anyway
            self._channel_finalizer.detach()

    async def exec_start(self, request: sr_pb2.TaskExecStartRequest) -> sr_pb2.TaskExecStartResponse:
        """Start an exec'd command, properly retrying on transient errors."""
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.TaskExecStart, request)
            )

    async def container_create(self, request: sr_pb2.TaskContainerCreateRequest) -> sr_pb2.TaskContainerCreateResponse:
        """Create an additional container via task command router."""
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.TaskContainerCreate, request)
            )

    async def container_terminate(
        self,
        request: sr_pb2.TaskContainerTerminateRequest,
    ) -> sr_pb2.TaskContainerTerminateResponse:
        """Terminate an additional container via task command router."""
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.TaskContainerTerminate, request)
            )

    async def container_wait(self, request: sr_pb2.TaskContainerWaitRequest) -> sr_pb2.TaskContainerWaitResponse:
        """Wait for an additional container via task command router."""
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.TaskContainerWait, request)
            )

    async def container_get(self, request: sr_pb2.TaskContainerGetRequest) -> sr_pb2.TaskContainerGetResponse:
        """Get the latest tracked container for a logical name via task command router."""
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.TaskContainerGet, request)
            )

    async def container_list(self, request: sr_pb2.TaskContainerListRequest) -> sr_pb2.TaskContainerListResponse:
        """List sandbox containers via task command router."""
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.TaskContainerList, request)
            )

    async def exec_stdio_read(
        self,
        task_id: str,
        exec_id: str,
        # Quotes around the type required for protobuf 3.19.
        file_descriptor: "api_pb2.FileDescriptor.ValueType",
        deadline: float | None = None,
    ) -> AsyncGenerator[sr_pb2.TaskExecStdioReadResponse, None]:
        """Stream stdout/stderr batches from an exec'd command, retrying on transient errors.

        Args:
            task_id: The task ID of the task running the exec'd command.
            exec_id: The execution ID of the command to read from.
            file_descriptor: The file descriptor to read from.
            deadline: The deadline by which all output must be streamed. If
              None, wait forever. If the deadline is exceeded, raises an
              ExecTimeoutError.
        Returns:
            AsyncGenerator[sr_pb2.TaskExecStdioReadResponse, None]: A stream of stdout/stderr batches.
        Raises:
            ExecTimeoutError: If the deadline is exceeded.
            Other errors: If retries are exhausted on transient errors or if there's an error
              from the RPC itself.
        """
        if file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
            sr_fd = sr_pb2.TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDOUT
        elif file_descriptor == api_pb2.FILE_DESCRIPTOR_STDERR:
            sr_fd = sr_pb2.TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDERR
        elif file_descriptor == api_pb2.FILE_DESCRIPTOR_INFO or file_descriptor == api_pb2.FILE_DESCRIPTOR_UNSPECIFIED:
            raise ValueError(f"Unsupported file descriptor: {file_descriptor}")
        else:
            raise ValueError(f"Invalid file descriptor: {file_descriptor}")

        with grpc_error_converter():
            async with aclosing(self._stream_stdio(task_id, exec_id, sr_fd, deadline)) as stream:
                async for item in stream:
                    yield item

    async def sandbox_stdio_read(
        self,
        task_id: str,
        # Quotes around the type required for protobuf 3.19.
        file_descriptor: "api_pb2.FileDescriptor.ValueType",
    ) -> AsyncGenerator[sr_pb2.SandboxStdioReadV2Response, None]:
        """Stream stdout/stderr batches from a V2 sandbox.

        Serves both live reads and post-exit reads.

        Args:
            task_id: The task ID hosting the V2 sandbox.
            file_descriptor: The file descriptor to read from (stdout or stderr).
        Returns:
            AsyncGenerator[sr_pb2.SandboxStdioReadV2Response, None]: A stream of stdout/stderr batches.
        Raises:
            Errors: If retries are exhausted on transient errors or if there is
              an error from the RPC itself.
        """
        if file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
            sr_fd = sr_pb2.SANDBOX_STDIO_FILE_DESCRIPTOR_STDOUT
        elif file_descriptor == api_pb2.FILE_DESCRIPTOR_STDERR:
            sr_fd = sr_pb2.SANDBOX_STDIO_FILE_DESCRIPTOR_STDERR
        elif file_descriptor == api_pb2.FILE_DESCRIPTOR_INFO or file_descriptor == api_pb2.FILE_DESCRIPTOR_UNSPECIFIED:
            raise ValueError(f"Unsupported file descriptor: {file_descriptor}")
        else:
            raise ValueError(f"Invalid file descriptor: {file_descriptor}")

        with grpc_error_converter():
            async with aclosing(self._stream_sandbox_stdio(task_id, sr_fd)) as stream:
                async for item in stream:
                    yield item

    async def exec_stdin_write(
        self, task_id: str, exec_id: str, offset: int, data: bytes, eof: bool
    ) -> sr_pb2.TaskExecStdinWriteResponse:
        """Write to the stdin stream of an exec'd command, properly retrying on transient errors.

        Args:
            task_id: The task ID of the task running the exec'd command.
            exec_id: The execution ID of the command to write to.
            offset: The offset to start writing to.
            data: The data to write to the stdin stream.
            eof: Whether to close the stdin stream after writing the data.
        Raises:
            Other errors: If retries are exhausted on transient errors or if there's an error
              from the RPC itself.
        """
        request = sr_pb2.TaskExecStdinWriteRequest(task_id=task_id, exec_id=exec_id, offset=offset, data=data, eof=eof)
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.TaskExecStdinWrite, request)
            )

    async def sandbox_stdin_write_v2(
        self, task_id: str, offset: int, data: bytes, eof: bool
    ) -> sr_pb2.SandboxStdinWriteV2Response:
        """Write to the stdin stream of a V2 sandbox's entrypoint process.

        Args:
            task_id: The task ID of the V2 sandbox.
            offset: The offset to start writing to.
            eof: Whether to close the stdin stream after writing the data.
        Raises:
            Other errors: If retries are exhausted on transient errors or if there's an error
              from the RPC itself.
        """
        request = sr_pb2.SandboxStdinWriteV2Request(task_id=task_id, offset=offset, data=data, eof=eof)
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.SandboxStdinWriteV2, request)
            )

    async def sandbox_wait_until_ready(self, task_id: str, timeout: float) -> sr_pb2.SandboxWaitUntilReadyTcrResponse:
        """Wait until the sandbox's readiness probe reports ready.

        Args:
            task_id: The task ID hosting the sandbox.
            timeout: Maximum time in seconds for the worker to wait.
        Raises:
            TimeoutError: If the sandbox does not become ready within `timeout`.
        """
        request = sr_pb2.SandboxWaitUntilReadyTcrRequest(task_id=task_id, timeout=timeout)
        with grpc_error_converter():
            try:
                return await asyncio.wait_for(
                    call_with_retries_on_transient_errors(
                        lambda: self._call_with_auth_retry(self._stub.SandboxWaitUntilReady, request, timeout=timeout),
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise ModalTimeoutError("Timeout expired")

    async def exec_poll(self, task_id: str, exec_id: str, deadline: float | None = None) -> sr_pb2.TaskExecPollResponse:
        """Poll for the exit status of an exec'd command, properly retrying on transient errors.

        Args:
            task_id: The task ID of the task running the exec'd command.
            exec_id: The execution ID of the command to poll on.
        Returns:
            sr_pb2.TaskExecPollResponse: The exit status of the command if it has completed.

        Raises:
            ExecTimeoutError: If the deadline is exceeded.
            Other errors: If retries are exhausted on transient errors or if there's an error
              from the RPC itself.
        """
        request = sr_pb2.TaskExecPollRequest(task_id=task_id, exec_id=exec_id)
        # The timeout here is really a backstop in the event of a hang contacting
        # the command router. Poll should usually be instantaneous.
        timeout = deadline - time.monotonic() if deadline is not None else None
        if timeout is not None and timeout <= 0:
            raise ExecTimeoutError(f"Deadline exceeded while polling for exec {exec_id}")

        with grpc_error_converter():
            try:
                return await asyncio.wait_for(
                    call_with_retries_on_transient_errors(
                        lambda: self._call_with_auth_retry(self._stub.TaskExecPoll, request)
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise ExecTimeoutError(f"Deadline exceeded while polling for exec {exec_id}")

    async def exec_wait(
        self,
        task_id: str,
        exec_id: str,
        deadline: float | None = None,
    ) -> sr_pb2.TaskExecWaitResponse:
        """Wait for an exec'd command to exit and return the exit code, properly retrying on transient errors.

        Args:
            task_id: The task ID of the task running the exec'd command.
            exec_id: The execution ID of the command to wait on.
        Returns:
            Optional[sr_pb2.TaskExecWaitResponse]: The exit code of the command.
        Raises:
            ExecTimeoutError: If the deadline is exceeded.
            Other errors: If there's an error from the RPC itself.
        """
        request = sr_pb2.TaskExecWaitRequest(task_id=task_id, exec_id=exec_id)
        timeout = deadline - time.monotonic() if deadline is not None else None
        if timeout is not None and timeout <= 0:
            raise ExecTimeoutError(f"Deadline exceeded while waiting for exec {exec_id}")

        with grpc_error_converter():
            try:
                return await asyncio.wait_for(
                    call_with_retries_on_transient_errors(
                        # We set a 60s timeout here to avoid waiting forever if there's an unanticipated hang
                        # due to a networking issue. call_with_retries_on_transient_errors will retry if the
                        # timeout is exceeded, so we'll retry every 60s until the command exits.
                        #
                        # Safety:
                        # * If just the task shuts down, the task command router will return a NOT_FOUND error,
                        #   and we'll stop retrying.
                        # * If the task shut down AND the worker shut down, this could
                        #   infinitely retry. For callers without an exec deadline, this
                        #   could hang indefinitely.
                        lambda: self._call_with_auth_retry(self._stub.TaskExecWait, request, timeout=60),
                        base_delay_secs=1,  # Retry after 1s since total time is expected to be long.
                        delay_factor=1,  # Fixed delay.
                        max_retries=None,  # Retry forever.
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise ExecTimeoutError(f"Deadline exceeded while waiting for exec {exec_id}")

    async def _refresh_jwt(self) -> None:
        """Refresh JWT from the server and update internal state."""
        async with self._jwt_refresh_lock:
            if self._closed:
                return

            # If the current JWT expiration is already far enough in the future, don't refresh.
            if self._jwt_exp is not None and self._jwt_exp - time.time() > 30:
                # This can happen if multiple concurrent requests to the task command router
                # get UNAUTHENTICATED errors and all refresh at the same time - one of them
                # will win and the others will not refresh.
                logger.debug(
                    f"Skipping JWT refresh for exec with task ID {self._task_id} "
                    "because its expiration is already far enough in the future"
                )
                return

            if self._is_v2_sandbox:
                logger.debug(f"Refreshing JWT for exec with sandbox ID {self._sandbox_id}")
                v2_resp = await fetch_command_router_access_v2(self._server_client, self._sandbox_id)
                logger.debug(f"Finished refreshing JWT for exec with sandbox ID {self._sandbox_id}")
                jwt, url = v2_resp.jwt, v2_resp.url
            else:
                logger.debug(f"Refreshing JWT for exec with task ID {self._task_id}")
                v1_resp = await fetch_command_router_access(self._server_client, self._task_id)
                logger.debug(f"Finished refreshing JWT for exec with task ID {self._task_id}")
                jwt, url = v1_resp.jwt, v1_resp.url

            # Ensure the server URL remains stable for the lifetime of this client.
            if url != self._server_url:
                logger.warning("Task router URL changed during session")
            self._jwt = jwt
            self._jwt_exp = _parse_jwt_expiration(jwt)

    async def _call_with_auth_retry(self, func, *args, **kwargs):
        try:
            return await func(*args, **kwargs, metadata=self._get_metadata())
        except GRPCError as exc:
            if exc.status == Status.UNAUTHENTICATED:
                await self._refresh_jwt()
                # Retry with the original arguments preserved
                return await func(*args, **kwargs, metadata=self._get_metadata())
            raise

    async def _stream_stdio_with_retries(
        self,
        *,
        stub_method: "grpclib.client.UnaryStreamMethod[_StdioReq, _StdioResp]",
        request_factory: Callable[[int], _StdioReq],
        deadline_label: str,
        deadline: float | None = None,
    ) -> AsyncGenerator[_StdioResp, None]:
        """Drive a streaming-stdio RPC with offset bookkeeping, transient-error
        retries, and JWT-refresh auth retries.

        Shared by [`_stream_stdio`] (exec stdio) and [`_stream_sandbox_stdio`]
        (V2 sandbox top-level stdio); both response types have a ``bytes data``
        field that this helper uses to advance the offset. For V2 sandbox
        responses (which carry ``starting_offset``), the offset is rebased off
        the first chunk of each attempt so transient reconnects don't miss
        bytes.
        """
        offset = 0
        delay_secs = self.stream_stdio_retry_delay_secs
        delay_factor = self.stream_stdio_retry_delay_factor
        num_retries_remaining = self.stream_stdio_max_retries
        # Flag to prevent infinite auth retries in the event that the JWT
        # refresh yields an invalid JWT somehow or that the JWT is otherwise invalid.
        did_auth_retry = False

        async def sleep_and_update_delay_and_num_retries_remaining(e: Exception):
            nonlocal delay_secs, num_retries_remaining
            logger.debug(f"Retrying stdio read with delay {delay_secs}s due to error: {e}")
            if deadline is not None and deadline - time.monotonic() <= delay_secs:
                raise ExecTimeoutError(f"Deadline exceeded while streaming stdio for {deadline_label}")

            await asyncio.sleep(delay_secs)
            delay_secs *= delay_factor
            num_retries_remaining -= 1

        while True:
            timeout = max(0, deadline - time.monotonic()) if deadline is not None else None
            try:
                stream = stub_method.open(timeout=timeout, metadata=self._get_metadata())
                async with stream as s:
                    req = request_factory(offset)

                    # Auth retry is scoped to a single refresh per streaming attempt. While auth metadata is
                    # sent on request start, UNAUTHENTICATED may sometimes surface during iteration,
                    # so we handle it at both send and receive boundaries.
                    is_first_chunk_of_attempt = True
                    try:
                        await s.send_message(req, end=True)
                        async for item in s:
                            # We successfully authenticated after a JWT refresh, reset the auth retry flag.
                            if did_auth_retry:
                                did_auth_retry = False
                            # Reset retry backoff after any successful chunk.
                            delay_secs = self.stream_stdio_retry_delay_secs
                            # Track it so transient reconnects request the
                            # correct next byte.
                            if is_first_chunk_of_attempt and isinstance(item, sr_pb2.SandboxStdioReadV2Response):
                                offset = item.starting_offset
                            is_first_chunk_of_attempt = False
                            offset += len(item.data)
                            yield item
                    except GRPCError as exc:
                        if exc.status == Status.UNAUTHENTICATED and not did_auth_retry:
                            await self._refresh_jwt()
                            # Mark that we've retried authentication for this streaming attempt, to
                            # prevent subsequent retries.
                            did_auth_retry = True
                            continue
                        raise

                # We successfully streamed all output.
                return
            except GRPCError as e:
                if num_retries_remaining > 0 and e.status in RETRYABLE_GRPC_STATUS_CODES:
                    await sleep_and_update_delay_and_num_retries_remaining(e)
                else:
                    raise e
            except AttributeError as e:
                # StreamTerminatedError are not properly raised in grpclib<=0.4.7
                # fixed in https://github.com/vmagamedov/grpclib/issues/185
                # TODO: update to newer version (>=0.4.8) once stable
                if num_retries_remaining > 0 and "_write_appdata" in str(e):
                    await sleep_and_update_delay_and_num_retries_remaining(e)
                else:
                    raise e
            except StreamTerminatedError as e:
                if num_retries_remaining > 0:
                    await sleep_and_update_delay_and_num_retries_remaining(e)
                else:
                    raise e
            except asyncio.TimeoutError as e:
                if num_retries_remaining > 0:
                    await sleep_and_update_delay_and_num_retries_remaining(e)
                else:
                    raise ConnectionError(str(e))
            except OSError as e:
                if num_retries_remaining > 0:
                    await sleep_and_update_delay_and_num_retries_remaining(e)
                else:
                    raise ConnectionError(str(e))

    async def _stream_stdio(
        self,
        task_id: str,
        exec_id: str,
        # Quotes around the type required for protobuf 3.19.
        file_descriptor: "sr_pb2.TaskExecStdioFileDescriptor.ValueType",
        deadline: float | None = None,
    ) -> AsyncGenerator[sr_pb2.TaskExecStdioReadResponse, None]:
        """Stream exec stdio from the task, retrying on transient errors.
        Raises ExecTimeoutError if the deadline is exceeded.
        """

        def request_factory(offset: int) -> sr_pb2.TaskExecStdioReadRequest:
            return sr_pb2.TaskExecStdioReadRequest(
                task_id=task_id,
                exec_id=exec_id,
                offset=offset,
                file_descriptor=file_descriptor,
            )

        async with aclosing(
            self._stream_stdio_with_retries(
                stub_method=self._stub.TaskExecStdioRead,
                request_factory=request_factory,
                deadline_label=f"exec {exec_id}",
                deadline=deadline,
            )
        ) as stream:
            async for item in stream:
                yield item

    async def _stream_sandbox_stdio(
        self,
        task_id: str,
        # Quotes around the type required for protobuf 3.19.
        file_descriptor: "sr_pb2.SandboxStdioFileDescriptor.ValueType",
    ) -> AsyncGenerator[sr_pb2.SandboxStdioReadV2Response, None]:
        """Stream V2 sandbox top-level stdio from the task, retrying on transient errors."""

        def request_factory(offset: int) -> sr_pb2.SandboxStdioReadV2Request:
            return sr_pb2.SandboxStdioReadV2Request(
                task_id=task_id,
                offset=offset,
                file_descriptor=file_descriptor,
            )

        async with aclosing(
            self._stream_stdio_with_retries(
                stub_method=self._stub.SandboxStdioReadV2,
                request_factory=request_factory,
                deadline_label=f"sandbox {task_id}",
                deadline=None,
            )
        ) as stream:
            async for item in stream:
                yield item

    async def mount_image(self, request: sr_pb2.TaskMountDirectoryRequest):
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.TaskMountDirectory, request)
            )

    async def unmount_image(self, request: sr_pb2.TaskUnmountDirectoryRequest):
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.TaskUnmountDirectory, request)
            )

    async def set_network_access(self, request: sr_pb2.TaskSetNetworkAccessRequest):
        with grpc_error_converter():
            return await call_with_retries_on_transient_errors(
                lambda: self._call_with_auth_retry(self._stub.TaskSetNetworkAccess, request)
            )

    async def reload_volumes(self, task_id: str, timeout: float) -> sr_pb2.TaskReloadVolumesResponse:
        """Reload all Volumes mounted in the task to reflect their latest committed state.

        Args:
            task_id: The task whose mounted Volumes should be reloaded.
            timeout: Client-side deadline in seconds. If the reload does not complete within this
                window, the call is cancelled and a `modal.exception.TimeoutError` is raised.
        """
        request = sr_pb2.TaskReloadVolumesRequest(task_id=task_id)
        with grpc_error_converter():
            try:
                return await asyncio.wait_for(
                    call_with_retries_on_transient_errors(
                        lambda: self._call_with_auth_retry(self._stub.TaskReloadVolumes, request, timeout=timeout),
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise ModalTimeoutError("Timeout expired")

    async def _snapshot_with_deadline(self, rpc, request, *, timeout: float, **kwargs):
        # helper method for snapshot_directory and snapshot_filesystem to handle grpc
        # deadlines in a consistent way, converting any error to TimeoutError after passing
        # the total deadline budget
        timeout_deadline = time.monotonic() + timeout

        def call():
            call_timeout = timeout_deadline - time.monotonic()
            if call_timeout <= 0.0:
                # doesn't matter which exception type this is
                # as it will be caught by the catch all below
                raise ModalTimeoutError("Timeout expired")
            return self._call_with_auth_retry(rpc, request, timeout=call_timeout, **kwargs)

        try:
            with grpc_error_converter():
                return await call_with_retries_on_transient_errors(
                    call,
                    exclude_status_codes=[Status.DEADLINE_EXCEEDED, Status.CANCELLED],
                    timeout_deadline=timeout_deadline,
                )
        except Exception:
            if time.monotonic() >= timeout_deadline:
                raise ModalTimeoutError("Timeout expired")
            raise

    async def snapshot_directory(
        self, request: sr_pb2.TaskSnapshotDirectoryRequest, *, timeout: float, **kwargs
    ) -> sr_pb2.TaskSnapshotDirectoryResponse:
        return await self._snapshot_with_deadline(self._stub.TaskSnapshotDirectory, request, timeout=timeout, **kwargs)

    async def snapshot_filesystem(
        self, request: sr_pb2.TaskSnapshotFilesystemRequest, *, timeout: float, **kwargs
    ) -> sr_pb2.TaskSnapshotFilesystemResponse:
        return await self._snapshot_with_deadline(self._stub.TaskSnapshotFilesystem, request, timeout=timeout, **kwargs)
