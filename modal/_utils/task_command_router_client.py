# Copyright Modal Labs 2025
import asyncio
import base64
import json
import ssl
import time
import urllib.parse
from typing import AsyncIterator, Optional

import grpclib.client
import grpclib.config
import grpclib.events
from grpclib import GRPCError, Status
from grpclib.exceptions import StreamTerminatedError

from modal.config import config, logger
from modal.exception import ExecTimeoutError
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2
from modal_proto.task_command_router_grpc import TaskCommandRouterStub

from .grpc_utils import RETRYABLE_GRPC_STATUS_CODES, connect_channel, retry_transient_errors


def _b64url_decode(data: str) -> bytes:
    """Decode a base64url string with missing padding tolerated."""
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _parse_jwt_expiration(jwt_token: str) -> Optional[float]:
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
    max_retries: Optional[int] = 10,
):
    """Call func() with transient error retries and exponential backoff.

    Authentication retries are expected to be handled by the caller.
    """
    delay_secs = base_delay_secs
    num_retries = 0

    async def sleep_and_update_delay_and_num_retries_remaining(e: Exception):
        nonlocal delay_secs, num_retries
        logger.debug(f"Retrying RPC with delay {delay_secs}s due to error: {e}")
        await asyncio.sleep(delay_secs)
        delay_secs *= delay_factor
        num_retries += 1

    while True:
        try:
            return await func()
        except GRPCError as e:
            if (max_retries is None or num_retries < max_retries) and e.status in RETRYABLE_GRPC_STATUS_CODES:
                await sleep_and_update_delay_and_num_retries_remaining(e)
            else:
                raise e
        except AttributeError as e:
            # StreamTerminatedError are not properly raised in grpclib<=0.4.7
            # fixed in https://github.com/vmagamedov/grpclib/issues/185
            # TODO: update to newer version (>=0.4.8) once stable
            if (max_retries is None or num_retries < max_retries) and "_write_appdata" in str(e):
                await sleep_and_update_delay_and_num_retries_remaining(e)
            else:
                raise e
        except StreamTerminatedError as e:
            if max_retries is None or num_retries < max_retries:
                await sleep_and_update_delay_and_num_retries_remaining(e)
            else:
                raise e
        except (OSError, asyncio.TimeoutError) as e:
            if max_retries is None or num_retries < max_retries:
                await sleep_and_update_delay_and_num_retries_remaining(e)
            else:
                raise ConnectionError(str(e))


async def fetch_command_router_access(server_client, task_id: str) -> api_pb2.TaskGetCommandRouterAccessResponse:
    """Fetch direct command router access info from Modal server."""
    return await retry_transient_errors(
        server_client.stub.TaskGetCommandRouterAccess,
        api_pb2.TaskGetCommandRouterAccessRequest(task_id=task_id),
    )


class TaskCommandRouterClient:
    """
    Client used to talk directly to TaskCommandRouter service on worker hosts.

    A new instance should be created per task.
    """

    @classmethod
    async def try_init(
        cls,
        server_client,
        task_id: str,
    ) -> Optional["TaskCommandRouterClient"]:
        """Attempt to initialize a TaskCommandRouterClient by fetching direct access.

        Returns None if command router access is not enabled (FAILED_PRECONDITION).
        """
        try:
            resp = await fetch_command_router_access(server_client, task_id)
        except GRPCError as exc:
            if exc.status == Status.FAILED_PRECONDITION:
                logger.debug(f"Command router access is not enabled for task {task_id}")
                return None
            raise

        logger.debug(f"Using command router access for task {task_id}")

        # Build and connect a channel to the task command router now that we have access info.
        o = urllib.parse.urlparse(resp.url)
        if o.scheme != "https":
            raise ValueError(f"Task router URL must be https, got: {resp.url}")

        host, _, port_str = o.netloc.partition(":")
        port = int(port_str) if port_str else 443
        ssl_context = ssl.create_default_context()

        # Allow insecure TLS when explicitly enabled via config.
        if config["task_command_router_insecure"]:
            logger.warning("Using insecure TLS for task command router due to MODAL_TASK_COMMAND_ROUTER_INSECURE")
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        channel = grpclib.client.Channel(
            host,
            port,
            ssl=ssl_context,
            config=grpclib.config.Configuration(
                http2_connection_window_size=64 * 1024 * 1024,  # 64 MiB
                http2_stream_window_size=64 * 1024 * 1024,  # 64 MiB
            ),
        )

        await connect_channel(channel)

        return cls(server_client, task_id, resp.url, resp.jwt, channel)

    def __init__(
        self,
        server_client,
        task_id: str,
        server_url: str,
        jwt: str,
        channel: grpclib.client.Channel,
        *,
        stream_stdio_retry_delay_secs: float = 0.01,
        stream_stdio_retry_delay_factor: float = 2,
        stream_stdio_max_retries: int = 10,
    ) -> None:
        """Callers should not use this directly. Use TaskCommandRouterClient.try_init() instead."""
        # Attach bearer token on all requests to the worker-side router service.
        self._server_client = server_client
        self._task_id = task_id
        self._server_url = server_url
        self._jwt = jwt
        self._channel = channel
        # Retry configuration for stdio streaming
        self.stream_stdio_retry_delay_secs = stream_stdio_retry_delay_secs
        self.stream_stdio_retry_delay_factor = stream_stdio_retry_delay_factor
        self.stream_stdio_max_retries = stream_stdio_max_retries

        # JWT refresh coordination
        self._jwt_exp: Optional[float] = _parse_jwt_expiration(jwt)
        self._jwt_refresh_lock = asyncio.Lock()
        self._jwt_refresh_event = asyncio.Event()
        self._closed = False

        # Start background task to eagerly refresh JWT 30s before expiration.
        self._jwt_refresh_task = asyncio.create_task(self._jwt_refresh_loop())

        async def send_request(event: grpclib.events.SendRequest) -> None:
            # This will get the most recent JWT for every request. No need to
            # lock _jwt_refresh_lock: reads and writes happen on the
            # single-threaded event loop and variable assignment is atomic.
            event.metadata["authorization"] = f"Bearer {self._jwt}"

        grpclib.events.listen(self._channel, grpclib.events.SendRequest, send_request)

        self._stub = TaskCommandRouterStub(self._channel)

    def __del__(self) -> None:
        """Clean up the client when it's garbage collected."""
        if self._closed:
            return

        self._jwt_refresh_task.cancel()

        try:
            self._channel.close()
        except Exception:
            pass

    async def close(self) -> None:
        """Close the client and stop the background JWT refresh task."""
        if self._closed:
            return

        self._closed = True
        self._jwt_refresh_task.cancel()
        try:
            logger.debug(f"Waiting for JWT refresh task to complete for exec with task ID {self._task_id}")
            await self._jwt_refresh_task
        except asyncio.CancelledError:
            pass
        self._channel.close()

    async def exec_start(self, request: sr_pb2.TaskExecStartRequest) -> sr_pb2.TaskExecStartResponse:
        """Start an exec'd command, properly retrying on transient errors."""
        return await call_with_retries_on_transient_errors(
            lambda: self._call_with_auth_retry(self._stub.TaskExecStart, request)
        )

    async def exec_stdio_read(
        self,
        task_id: str,
        exec_id: str,
        # Quotes around the type required for protobuf 3.19.
        file_descriptor: "api_pb2.FileDescriptor.ValueType",
        deadline: Optional[float] = None,
    ) -> AsyncIterator[sr_pb2.TaskExecStdioReadResponse]:
        """Stream stdout/stderr batches from the task, properly retrying on transient errors.

        Args:
            task_id: The task ID of the task running the exec'd command.
            exec_id: The execution ID of the command to read from.
            file_descriptor: The file descriptor to read from.
            deadline: The deadline by which all output must be streamed. If
              None, wait forever. If the deadline is exceeded, raises an
              ExecTimeoutError.
        Returns:
            AsyncIterator[sr_pb2.TaskExecStdioReadResponse]: A stream of stdout/stderr batches.
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

        async for item in self._stream_stdio(task_id, exec_id, sr_fd, deadline):
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
        return await call_with_retries_on_transient_errors(
            lambda: self._call_with_auth_retry(self._stub.TaskExecStdinWrite, request)
        )

    async def exec_poll(
        self, task_id: str, exec_id: str, deadline: Optional[float] = None
    ) -> sr_pb2.TaskExecPollResponse:
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
        deadline: Optional[float] = None,
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
        """Refresh JWT from the server and update internal state.

        Concurrency-safe: only one refresh runs at a time.
        """
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

            resp = await fetch_command_router_access(self._server_client, self._task_id)
            # Ensure the server URL remains stable for the lifetime of this client.
            assert resp.url == self._server_url, "Task router URL changed during session"
            self._jwt = resp.jwt
            self._jwt_exp = _parse_jwt_expiration(resp.jwt)
            # Wake up the background loop to recompute its next sleep.
            self._jwt_refresh_event.set()

    async def _call_with_auth_retry(self, func, *args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except GRPCError as exc:
            if exc.status == Status.UNAUTHENTICATED:
                await self._refresh_jwt()
                # Retry with the original arguments preserved
                return await func(*args, **kwargs)
            raise

    async def _jwt_refresh_loop(self) -> None:
        """Background task that refreshes JWT 30 seconds before expiration.

        Uses an event to wake early when a manual refresh happens or token changes.
        """
        while not self._closed:
            try:
                exp = self._jwt_exp
                now = time.time()
                if exp is None:
                    # Unknown expiration: re-check periodically or until event wakes us.
                    sleep_s = 60.0
                else:
                    refresh_at = exp - 30.0
                    sleep_s = max(refresh_at - now, 0.0)

                self._jwt_refresh_event.clear()
                if sleep_s > 0:
                    try:
                        logger.debug(f"Waiting for JWT refresh for {sleep_s}s for exec with task ID {self._task_id}")
                        # Wait until it's time to refresh, unless woken early.
                        await asyncio.wait_for(self._jwt_refresh_event.wait(), timeout=sleep_s)
                        logger.debug(f"Stopped waiting for JWT refresh for exec with task ID {self._task_id}")
                        # Event fired (e.g., token changed) -> recompute timings.
                        continue
                    except asyncio.TimeoutError:
                        logger.debug(f"Done waiting for JWT refresh for exec with task ID {self._task_id}")
                        pass

                # Time to refresh.
                logger.debug(f"Refreshing JWT for exec with task ID {self._task_id}")
                await self._refresh_jwt()
            except asyncio.CancelledError:
                logger.debug(f"Cancelled JWT refresh loop for exec with task ID {self._task_id}")
                break
            except Exception as e:
                logger.warning(f"Background JWT refresh failed for exec with task ID {self._task_id}: {e}")
                try:
                    await asyncio.sleep(1.0)
                except Exception:
                    # Ignore sleep issues; loop will re-check closed flag.
                    pass

    async def _stream_stdio(
        self,
        task_id: str,
        exec_id: str,
        # Quotes around the type required for protobuf 3.19.
        file_descriptor: "sr_pb2.TaskExecStdioFileDescriptor.ValueType",
        deadline: Optional[float] = None,
    ) -> AsyncIterator[sr_pb2.TaskExecStdioReadResponse]:
        """Stream stdio from the task, properly updating the offset and retrying on transient errors.
        Raises ExecTimeoutError if the deadline is exceeded.
        """
        offset = 0
        delay_secs = self.stream_stdio_retry_delay_secs
        delay_factor = self.stream_stdio_retry_delay_factor
        num_retries_remaining = self.stream_stdio_max_retries
        num_auth_retries = 0

        async def sleep_and_update_delay_and_num_retries_remaining(e: Exception):
            nonlocal delay_secs, num_retries_remaining
            logger.debug(f"Retrying stdio read with delay {delay_secs}s due to error: {e}")
            if deadline is not None and deadline - time.monotonic() <= delay_secs:
                raise ExecTimeoutError(f"Deadline exceeded while streaming stdio for exec {exec_id}")

            await asyncio.sleep(delay_secs)
            delay_secs *= delay_factor
            num_retries_remaining -= 1

        while True:
            timeout = max(0, deadline - time.monotonic()) if deadline is not None else None
            try:
                stream = self._stub.TaskExecStdioRead.open(timeout=timeout)
                async with stream as s:
                    req = sr_pb2.TaskExecStdioReadRequest(
                        task_id=task_id,
                        exec_id=exec_id,
                        offset=offset,
                        file_descriptor=file_descriptor,
                    )

                    # Scope auth retry strictly to the initial send (where headers/auth are sent).
                    try:
                        await s.send_message(req, end=True)
                    except GRPCError as exc:
                        if exc.status == Status.UNAUTHENTICATED and num_auth_retries < 1:
                            await self._refresh_jwt()
                            num_auth_retries += 1
                            continue
                        raise

                    # We successfully authenticated, reset the auth retry count.
                    num_auth_retries = 0

                    async for item in s:
                        # Reset retry backoff after any successful chunk.
                        delay_secs = self.stream_stdio_retry_delay_secs
                        offset += len(item.data)
                        yield item

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
