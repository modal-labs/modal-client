# Copyright Modal Labs 2025
import asyncio
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
from modal_proto import api_pb2, sandbox_router_pb2 as sr_pb2
from modal_proto.sandbox_router_grpc import SandboxRouterStub

from .grpc_utils import RETRYABLE_GRPC_STATUS_CODES, connect_channel, retry_transient_errors


async def call_with_retries_on_transient_errors(
    func,
    *,
    base_delay: float = 0.01,
    delay_factor: float = 2,
    max_retries: Optional[int] = 10,
):
    """Call func() with transient error retries and exponential backoff.

    Authentication retries are expected to be handled by the caller.
    """
    delay = base_delay
    num_retries = 0

    async def sleep_and_update_delay_and_num_retries_remaining(e: Exception):
        nonlocal delay, num_retries
        logger.debug(f"Retrying RPC with delay {delay}s due to error: {e}")
        await asyncio.sleep(delay)
        delay *= delay_factor
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
            # TODO(saltzm): update to newer version (>=0.4.8) once stable
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


async def fetch_command_router_access(server_client, sandbox_id: str) -> api_pb2.SandboxGetCommandRouterAccessResponse:
    """Fetch direct command router access info from Modal server."""
    return await retry_transient_errors(
        server_client.stub.SandboxGetCommandRouterAccess,
        api_pb2.SandboxGetCommandRouterAccessRequest(sandbox_id=sandbox_id),
    )


class SandboxRouterClient:
    """
    Client used to talk directly to SandboxRouterService on worker hosts.

    A new instance should be created per sandbox.
    """

    def __init__(
        self,
        server_client,
        sandbox_id: str,
        server_url: str,
        jwt: str,
        *,
        stream_stdio_retry_delay: float = 0.01,
        stream_stdio_retry_delay_factor: float = 2,
        stream_stdio_max_retries: int = 10,
    ) -> None:
        """Callers should not use this directly. Use SandboxRouterClient.try_init() instead."""
        # Attach bearer token on all requests to the worker-side router service.
        self._server_client = server_client
        self._sandbox_id = sandbox_id
        self._jwt = jwt
        self._server_url = server_url
        # Retry configuration for stdio streaming
        self.stream_stdio_retry_delay = stream_stdio_retry_delay
        self.stream_stdio_retry_delay_factor = stream_stdio_retry_delay_factor
        self.stream_stdio_max_retries = stream_stdio_max_retries

        # Only https URLs are supported for the sandbox router. Build a channel with a TLS context.
        o = urllib.parse.urlparse(server_url)
        if o.scheme != "https":
            raise ValueError(f"Sandbox router URL must be https, got: {server_url}")

        host, _, port_str = o.netloc.partition(":")
        port = int(port_str) if port_str else 443
        ssl_context = ssl.create_default_context()

        # Allow insecure TLS when explicitly enabled via config (respects env
        # var MODAL_SANDBOX_ROUTER_INSECURE). Used for local testing.
        #
        # TODO(saltzm): There's probably a better way to do this using a local
        # cert. Seek PR feedback.
        if config["sandbox_router_insecure"]:
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

        async def send_request(event: grpclib.events.SendRequest) -> None:
            # This will get the most recent JWT for every request.
            event.metadata["authorization"] = f"Bearer {self._jwt}"

        grpclib.events.listen(channel, grpclib.events.SendRequest, send_request)

        self._channel = channel

        # Don't access this directly, use _get_stub() instead.
        self._stub = SandboxRouterStub(self._channel)
        self._connected = False

    @classmethod
    async def try_init(
        cls,
        server_client,
        sandbox_id: str,
    ) -> Optional["SandboxRouterClient"]:
        """Attempt to initialize a SandboxRouterClient by fetching direct access.

        Returns None if command router access is not enabled (FAILED_PRECONDITION).
        """
        try:
            resp = await fetch_command_router_access(server_client, sandbox_id)
        except GRPCError as exc:
            if exc.status == Status.FAILED_PRECONDITION:
                return None
            raise

        return cls(server_client, sandbox_id, resp.url, resp.jwt)

    async def _refresh_jwt(self) -> None:
        # TODO(saltzm): This is inefficient in the event that multiple execs are issued concurrently,
        # since each concurrent request will fetch a new token. It doesn't matter for correctness
        # because the last written token will still be valid and usable by all execs, but it's
        # confusing and inefficient. We need to fix this.
        resp = await fetch_command_router_access(self._server_client, self._sandbox_id)
        # Ensure the server URL remains stable for the lifetime of this client
        assert resp.url == self._server_url, "Sandbox router URL changed during session"
        self._jwt = resp.jwt

    async def _call_with_auth_retry(self, func):
        try:
            return await func()
        except GRPCError as exc:
            if exc.status == Status.UNAUTHENTICATED:
                await self._refresh_jwt()
                return await func()
            raise

    def __del__(self) -> None:
        # Best-effort cleanup to avoid noisy "Unclosed connection" warnings.
        try:
            self._channel.close()
        except Exception:
            # Avoid raising during interpreter shutdown.
            pass

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await connect_channel(self._channel)
            self._connected = True

    async def _get_stub(self) -> SandboxRouterStub:
        await self._ensure_connected()
        return self._stub

    async def close(self) -> None:
        self._channel.close()
        self._connected = False

    async def exec_start(self, request: sr_pb2.SandboxExecStartRequest) -> sr_pb2.SandboxExecStartResponse:
        stub = await self._get_stub()
        return await call_with_retries_on_transient_errors(
            lambda: self._call_with_auth_retry(lambda: stub.SandboxExecStart(request))
        )

    async def exec_stdio_read(
        self,
        task_id: str,
        exec_id: str,
        # Quotes around the type required for protobuf 3.19.
        file_descriptor: "api_pb2.FileDescriptor.ValueType",
        deadline: Optional[float] = None,
    ) -> AsyncIterator[sr_pb2.SandboxExecStdioReadResponse]:
        """Stream stdout/stderr batches from the sandbox, properly retrying on transient errors.

        Args:
            task_id: The task ID of the sandbox running the exec'd command.
            exec_id: The execution ID of the command to read from.
            file_descriptor: The file descriptor to read from.
            deadline: The deadline by which all output must be streamed. If
              None, wait forever. If the deadline is exceeded, raises an
              ExecTimeoutError.
        Returns:
            AsyncIterator[sr_pb2.SandboxExecStdioReadResponse]: A stream of stdout/stderr batches.
        Raises:
            ExecTimeoutError: If the deadline is exceeded.
            Other errors: If retries are exhausted on transient errors or if there's an error
              from the RPC itself.
        """
        if file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
            sr_fd = sr_pb2.SANDBOX_EXEC_STDIO_FILE_DESCRIPTOR_STDOUT
        elif file_descriptor == api_pb2.FILE_DESCRIPTOR_STDERR:
            sr_fd = sr_pb2.SANDBOX_EXEC_STDIO_FILE_DESCRIPTOR_STDERR
        elif file_descriptor == api_pb2.FILE_DESCRIPTOR_INFO or file_descriptor == api_pb2.FILE_DESCRIPTOR_UNSPECIFIED:
            raise ValueError(f"Unsupported file descriptor: {file_descriptor}")
        else:
            raise ValueError(f"Invalid file descriptor: {file_descriptor}")

        async for item in self._stream_stdio(task_id, exec_id, sr_fd, deadline):
            yield item

    async def _stream_stdio(
        self,
        task_id: str,
        exec_id: str,
        # Quotes around the type required for protobuf 3.19.
        file_descriptor: "sr_pb2.SandboxExecStdioFileDescriptor.ValueType",
        deadline: Optional[float] = None,
    ) -> AsyncIterator[sr_pb2.SandboxExecStdioReadResponse]:
        """Stream stdio from the sandbox, properly updating the offset and retrying on transient errors.
        Raises ExecTimeoutError if the deadline is exceeded.
        """
        offset = 0
        delay = self.stream_stdio_retry_delay
        delay_factor = self.stream_stdio_retry_delay_factor
        num_retries_remaining = self.stream_stdio_max_retries
        num_auth_retries = 0

        async def sleep_and_update_delay_and_num_retries_remaining(e: Exception):
            """Helper for retrying on transient errors."""
            nonlocal delay, num_retries_remaining
            logger.debug(f"Retrying stdio read with delay {delay}s due to error: {e}")
            await asyncio.sleep(delay)
            delay *= delay_factor
            num_retries_remaining -= 1

        while True:
            stub = await self._get_stub()
            timeout = deadline - time.monotonic() if deadline is not None else None
            if timeout is not None and timeout <= 0:
                raise ExecTimeoutError(f"Deadline exceeded while streaming stdio for exec {exec_id}")
            stream = stub.SandboxExecStdioRead.open(timeout=timeout)
            try:
                async with stream as s:
                    req = sr_pb2.SandboxExecStdioReadRequest(
                        task_id=task_id,
                        exec_id=exec_id,
                        offset=offset,
                        file_descriptor=file_descriptor,
                    )

                    # Scope retry strictly to the initial send (where headers/auth are sent)
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
                        offset += len(item.data)
                        yield item

                # We successfully streamed all output.
                return
            except GRPCError as e:
                # Check if we would have exceeded the deadline by the time we retry.
                deadline_exceeded = deadline is not None and time.monotonic() + delay >= deadline
                if num_retries_remaining > 0 and e.status in RETRYABLE_GRPC_STATUS_CODES and not deadline_exceeded:
                    await sleep_and_update_delay_and_num_retries_remaining(e)
                elif deadline_exceeded:
                    raise ExecTimeoutError(f"Deadline exceeded while streaming stdio for exec {exec_id}")
                else:
                    raise e
            except AttributeError as e:
                # StreamTerminatedError are not properly raised in grpclib<=0.4.7
                # fixed in https://github.com/vmagamedov/grpclib/issues/185
                # TODO(saltzm): update to newer version (>=0.4.8) once stable
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
                # Check if we would have exceeded the deadline by the time we retry.
                deadline_exceeded = deadline is not None and time.monotonic() + delay >= deadline
                if num_retries_remaining > 0 and not deadline_exceeded:
                    await sleep_and_update_delay_and_num_retries_remaining(e)
                elif deadline_exceeded:
                    raise ExecTimeoutError(f"Deadline exceeded while streaming stdio for exec {exec_id}")
                else:
                    raise ConnectionError(str(e))
            except OSError as e:
                if num_retries_remaining > 0:
                    await sleep_and_update_delay_and_num_retries_remaining(e)
                else:
                    raise ConnectionError(str(e))

    async def exec_stdin_write(
        self, task_id: str, exec_id: str, offset: int, data: bytes, eof: bool
    ) -> sr_pb2.SandboxExecStdinWriteResponse:
        """Write to the stdin stream of an exec'd command, properly retrying on transient errors.

        Args:
            task_id: The task ID of the sandbox running the exec'd command.
            exec_id: The execution ID of the command to write to.
            offset: The offset to start writing to.
            data: The data to write to the stdin stream.
            eof: Whether to close the stdin stream after writing the data.
        Raises:
            Other errors: If retries are exhausted on transient errors or if there's an error
              from the RPC itself.
        """
        stub = await self._get_stub()
        request = sr_pb2.SandboxExecStdinWriteRequest(
            task_id=task_id, exec_id=exec_id, offset=offset, data=data, eof=eof
        )
        return await call_with_retries_on_transient_errors(
            lambda: self._call_with_auth_retry(lambda: stub.SandboxExecStdinWrite(request))
        )

    async def exec_poll(self, task_id: str, exec_id: str) -> sr_pb2.SandboxExecPollResponse:
        """Poll for the exit status of an exec'd command, properly retrying on transient errors.

        Args:
            task_id: The task ID of the sandbox running the exec'd command.
            exec_id: The execution ID of the command to poll on.
        Returns:
            sr_pb2.SandboxExecPollResponse: The exit status of the command if it has completed.

        Raises:
            Other errors: If retries are exhausted on transient errors or if there's an error
              from the RPC itself.
        """
        stub = await self._get_stub()
        request = sr_pb2.SandboxExecPollRequest(task_id=task_id, exec_id=exec_id)
        return await call_with_retries_on_transient_errors(
            lambda: self._call_with_auth_retry(lambda: stub.SandboxExecPoll(request))
        )

    async def exec_wait(
        self,
        task_id: str,
        exec_id: str,
        deadline: Optional[float] = None,
    ) -> sr_pb2.SandboxExecWaitResponse:
        """Wait for an exec'd command to exit and return the exit code, properly retrying on transient errors.

        Args:
            task_id: The task ID of the sandbox running the exec'd command.
            exec_id: The execution ID of the command to wait on.
        Returns:
            Optional[sr_pb2.SandboxExecWaitResponse]: The exit code of the command.
        Raises:
            ExecTimeoutError: If the deadline is exceeded.
            Other errors: If there's an error from the RPC itself.
        """
        stub = await self._get_stub()
        request = sr_pb2.SandboxExecWaitRequest(task_id=task_id, exec_id=exec_id)
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
                    # * If just the sandbox shuts down, the sandbox router will return a NOT_FOUND error,
                    #   and we'll stop retrying.
                    # * If the sandbox shut down AND the worker shut down, this could
                    #   infinitely retry. For callers without an exec deadline, this
                    #   could hang indefinitely.
                    lambda: self._call_with_auth_retry(lambda: stub.SandboxExecWait(request, timeout=60)),
                    base_delay=1,  # Retry after 1s since total time is expected to be long.
                    delay_factor=1,  # Fixed delay.
                    max_retries=None,  # Retry forever.
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise ExecTimeoutError(f"Deadline exceeded while waiting for exec {exec_id}")
