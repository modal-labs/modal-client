# Copyright Modal Labs 2026
import asyncio
import json
import re
import time
import urllib.parse
import uuid
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from synchronicity import classproperty

from .._environments import _Environment
from .._object import _get_environment_name
from .._utils.async_utils import synchronize_api
from .._utils.http_utils import ClientSessionRegistry
from ..client import _Client
from ..config import DEFAULT_STREAMS_URL, config, logger
from ..exception import (
    AlreadyExistsError,
    AuthError,
    ConflictError,
    InvalidError,
    NotFoundError,
    PermissionDeniedError,
    ServiceError,
)

ENVIRONMENT_HEADER = "x-streams-environment"
# URL-safe characters only, since names appear in URL paths; 64 matches other Modal object names.
STREAM_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
# The service's per-value limit, checked here so an oversized value fails before it is sent.
MAX_VALUE_BYTES = 64 * 1024
# The service expires streams in an hourly pass, so a shorter retention would not be honored.
MIN_RETENTION = timedelta(hours=1)
# Generous for one append or one page of reads; a request that takes longer is retried.
REQUEST_TIMEOUT_S = 30.0
# Followed reads receive a keep-alive every 15 s, so only a dead connection goes this long without data.
FOLLOW_IDLE_TIMEOUT_S = 60.0
# Transient failures are retried for this long, which covers the streams service recovering from a full restart.
RETRY_DEADLINE_S = 60.0
# Waits between retries double from 0.1 s up to 5 s.
RETRY_BASE_DELAY_S = 0.1
RETRY_MAX_DELAY_S = 5.0
# Pause before reopening a follow the service ended cleanly, so one that ends follows at once is not hammered.
FOLLOW_REOPEN_DELAY_S = 0.1
# Writes rejected because another writer took over the stream are resent under its lease at most this many times.
FENCED_RETRIES = 10
# The most entries the service returns per read request.
READ_PAGE_SIZE = 1000
# Values per write request, which keeps a request to a few MiB at the largest values.
WRITER_MAX_BATCH = 100
# Values a writer queues before `append` waits, at most about 64 MiB of serialized JSON at the largest values.
WRITER_MAX_PENDING = 1000


@dataclass(frozen=True)
class StreamInfo:
    """A Stream's state: `last_seq` is the sequence number of its latest entry, or 0 while it is empty."""

    name: str
    last_seq: int
    sealed: bool
    created_at: datetime
    last_written_at: datetime
    expires_at: datetime | None


@dataclass(frozen=True)
class AppendResult:
    """Where an appended value landed; `deduped` when its idempotency key had already been written."""

    seq: int
    deduped: bool


@dataclass(frozen=True)
class Entry:
    """One entry read from a Stream: a value, or the final `seal` entry."""

    seq: int
    idempotency_key: str
    kind: Literal["value", "seal"]
    value: Any
    ts: datetime


class IdempotencyConflict(ConflictError):
    """An idempotency key was reused with a different value; `seq` is where the original value landed."""

    idempotency_key: str | None
    seq: int | None

    def __init__(self, message: str, *, idempotency_key: str | None, seq: int | None):
        super().__init__(message)
        self.idempotency_key = idempotency_key
        self.seq = seq


class _Transient(ServiceError):
    pass


class _Fenced(ConflictError):
    pass


def _service_url() -> str:
    url = config.get("streams_url")
    if not url or url == DEFAULT_STREAMS_URL:
        return DEFAULT_STREAMS_URL
    # Requests carry Modal credentials, so they may only be redirected to a service on this machine.
    if urllib.parse.urlsplit(url).hostname not in ("localhost", "127.0.0.1", "::1"):
        raise InvalidError("`MODAL_STREAMS_URL` can only point at a streams service on localhost.")
    return url.rstrip("/")


def _backoff(attempt: int) -> float:
    return min(RETRY_BASE_DELAY_S * 2 ** (attempt - 1), RETRY_MAX_DELAY_S)


def _check_name(name: str) -> None:
    if not STREAM_NAME.fullmatch(name):
        raise InvalidError(
            f"Invalid Stream name: '{name}'. Names must be 1-64 alphanumeric characters, dashes, and underscores."
        )


def _check_retention(retention: timedelta | None) -> None:
    if retention is not None and retention < MIN_RETENTION:
        raise InvalidError("`retention` must be at least one hour.")


def _error_for(status: int, body: bytes, stream_name: str) -> Exception:
    err: dict[str, Any] = {}
    try:
        err = json.loads(body)["error"]
    except (ValueError, KeyError, TypeError):
        pass
    code: str = err.get("code") or ""
    message = f"Stream '{stream_name}': {err.get('message') or body.decode(errors='replace')[:500] or status}"
    if status in (429, 500, 502, 503, 504):
        return _Transient(message)
    if status == 409:
        if code == "idempotency_conflict":
            return IdempotencyConflict(message, idempotency_key=err.get("op_id"), seq=err.get("seq"))
        return {"already_exists": AlreadyExistsError, "fenced": _Fenced}.get(code, ConflictError)(message)
    errors = {400: InvalidError, 413: InvalidError, 401: AuthError, 403: PermissionDeniedError, 404: NotFoundError}
    return errors.get(status, NotFoundError if status == 410 else ServiceError)(message)


def _check_value(value: Any) -> None:
    try:
        size = len(json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())
    except (TypeError, ValueError) as exc:
        raise InvalidError(f"Stream values must be JSON-serializable: {exc}") from None
    if size > MAX_VALUE_BYTES:
        raise InvalidError(f"Stream values are limited to {MAX_VALUE_BYTES} bytes of JSON; this one is {size} bytes.")


def _entry(event: dict[str, Any]) -> Entry:
    return Entry(
        seq=int(event["seq"]),
        idempotency_key=event.get("op_id", ""),
        kind=event.get("kind", "value"),
        value=event.get("value"),
        ts=_utc(event.get("ts_us", 0)),
    )


def _utc(ts_us: int) -> datetime:
    return datetime.fromtimestamp(ts_us / 1_000_000, tz=timezone.utc)


async def _sse_messages(content: Any) -> AsyncGenerator[tuple[str, str], None]:
    """The (event, data) of each server-sent event; lines are split by hand since values can exceed
    aiohttp's 64 KiB line limit."""
    event, data, pending = "message", [], b""
    async for chunk in content.iter_any():
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for raw in lines:
            line = raw.decode().removesuffix("\r")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
            elif not line:
                if data or event != "message":
                    yield event, "\n".join(data)
                event, data = "message", []


class _StreamManager:
    """Namespace with methods for managing named Streams."""

    async def delete(
        self,
        name: str,
        *,
        allow_missing: bool = False,
        environment_name: str | None = None,
        client: _Client | None = None,
    ) -> None:
        """Delete a named Stream and all of its entries.

        Deletion is irreversible; handles to the Stream then raise `NotFoundError`.

        Args:
            name: Name of the Stream to delete.
            allow_missing: If True, do nothing when the Stream does not exist.
            environment_name: Environment to delete from; defaults to the active environment.
            client: Modal client to use; defaults to `Client.from_env()` when omitted.

        Examples:
            ```python notest
            modal.experimental.Stream.objects.delete("my-stream")
            ```
        """
        stream = _Stream.from_name(name, environment_name=environment_name, client=client)
        try:
            # Not retried: a resent delete could remove a stream re-created under the same name meanwhile.
            await stream._call("DELETE", stream._path, retry=False)
        except NotFoundError:
            if not allow_missing:
                raise


StreamManager = synchronize_api(_StreamManager, target_module=__name__)


class _Stream:
    """A named, durable, append-only stream of JSON values.

    Each value gets a dense sequence number starting at 1. Readers can replay a stream from any point
    and follow it for new values until it is sealed. Any number of handles, in any
    process, may append to the same stream.

    Streams are in limited preview: other workspaces get a `PermissionDeniedError`.

    **Usage**

    ```python notest
    import modal

    stream = modal.experimental.Stream.from_name("my-stream", create_if_missing=True)
    stream.append({"role": "user", "content": "hi"}, idempotency_key="turn-1")
    for entry in stream.read(follow=True):
        print(entry.seq, entry.value)
    ```
    """

    def __init__(
        self,
        name: str,
        environment_name: str,
        create_if_missing: bool,
        retention: timedelta | None,
        client: _Client | None,
    ):
        """mdmd:hidden"""
        self.name = name
        self._path = f"/v1/runs/{name}"
        self._environment_name = environment_name
        self._create_if_missing = create_if_missing
        self._retention = retention
        self._client = client
        self._gen: int | None = None
        self._lock = asyncio.Lock()

    @classproperty
    @classmethod
    def objects(cls) -> _StreamManager:
        return _StreamManager()

    @staticmethod
    def from_name(
        name: str,
        *,
        environment_name: str | None = None,
        create_if_missing: bool = False,
        retention: timedelta | None = None,
        client: _Client | None = None,
    ) -> "_Stream":
        """Reference a Stream by name; nothing is sent until its first use, which creates it if asked.

        `retention` sets how long a stream this call creates is kept after its last write, from one hour
        up to the service's maximum; without it the service's default applies. It has no effect on a
        stream that already exists.
        """
        _check_retention(retention)
        _check_name(name)
        return _Stream(name, _get_environment_name(environment_name), create_if_missing, retention, client)

    async def _headers(self) -> dict[str, str]:
        if self._client is None:
            self._client = await _Client.from_env()
        assert self._client._auth_token_manager, "Client is not open"
        token = await self._client._auth_token_manager.get_token()
        if not self._environment_name:
            self._environment_name = (await _Environment.from_context(client=self._client).hydrate()).name or ""
        return {"Authorization": f"Bearer {token}", ENVIRONMENT_HEADER: self._environment_name}

    async def _call(
        self,
        method: str,
        path: str,
        body: Any = None,
        params: Mapping[str, Any] | None = None,
        retry: bool = True,
    ) -> tuple[bytes, Mapping[str, str]]:
        """One request, retried on transient failures until `RETRY_DEADLINE_S` passes, if `retry`."""
        import aiohttp

        deadline = time.monotonic() + RETRY_DEADLINE_S
        attempt = 0
        while True:
            attempt += 1
            try:
                async with ClientSessionRegistry.get_session().request(
                    method,
                    _service_url() + path,
                    json=body,
                    params=params,
                    headers=await self._headers(),
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
                ) as resp:
                    data = await resp.read()
                    if resp.status < 400:
                        return data, resp.headers
                    raise _error_for(resp.status, data, self.name)
            except (aiohttp.ClientError, asyncio.TimeoutError, _Transient) as exc:
                delay = _backoff(attempt)
                if not retry or time.monotonic() + delay > deadline:
                    raise self._unreachable(exc) from exc
                await asyncio.sleep(delay)

    def _unreachable(self, exc: Exception) -> Exception:
        if isinstance(exc, _Transient):
            return exc
        return ServiceError(f"Stream '{self.name}': could not reach the streams service: {exc!r}")

    async def _create_locked(self) -> None:
        """Create the stream on this handle's first request if `create_if_missing` was set; the caller
        holds `_lock`, so concurrent first requests create it once."""
        if self._create_if_missing:
            body: dict[str, Any] = {"run_id": self.name}
            if self._retention is not None:
                body["retention_s"] = int(self._retention.total_seconds())
            try:
                data, _ = await self._call("POST", "/v1/runs", body)
                self._gen = json.loads(data)["gen"]
            except AlreadyExistsError:
                pass
            self._create_if_missing = False

    async def _lease(self) -> int:
        """The lease number this client attaches to its writes. The service rejects a stale one, and since
        this client never takes a new lease, all handles share the current one."""
        async with self._lock:
            await self._create_locked()
            gen = self._gen
            if gen is None:
                data, _ = await self._call("GET", self._path)
                gen = self._gen = json.loads(data)["gen"]
            return gen

    async def _write(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """A write under the stream's current generation, resent under the new one if it changes."""
        for attempt in range(1, FENCED_RETRIES + 1):
            gen = await self._lease()
            try:
                data, _ = await self._call("POST", self._path + path, {**body, "gen": gen})
                return json.loads(data)
            except _Fenced as exc:
                async with self._lock:
                    if self._gen == gen:
                        self._gen = None
                if attempt == FENCED_RETRIES:
                    raise ConflictError(str(exc)) from None
                await asyncio.sleep(_backoff(attempt))
        raise AssertionError("unreachable")

    async def append(self, value: Any, *, idempotency_key: str | None = None) -> AppendResult:
        """Append a JSON-serializable value of at most 64 KiB and return its sequence number.

        Appending again with the same `idempotency_key` and value returns the original result with
        `deduped=True`; the same key with a different value raises `IdempotencyConflict`.
        """
        keys = None if idempotency_key is None else [idempotency_key]
        return (await self.append_many([value], idempotency_keys=keys))[0]

    async def append_many(
        self, values: Iterable[Any], *, idempotency_keys: Sequence[str] | None = None
    ) -> list[AppendResult]:
        """Append several values in one request, in order; see `append`. If any key conflicts, none
        of the values is written."""
        values = list(values)
        if not values:
            return []
        if idempotency_keys is None:
            idempotency_keys = [uuid.uuid4().hex for _ in values]
        elif len(idempotency_keys) != len(values):
            raise InvalidError("`idempotency_keys` must have one key per value.")
        if len(set(idempotency_keys)) != len(idempotency_keys):
            raise InvalidError("`idempotency_keys` must be distinct within one call.")
        for value in values:
            _check_value(value)
        events = [{"op_id": k, "kind": "value", "value": v} for k, v in zip(idempotency_keys, values)]
        out = await self._write("/append", {"events": events})
        return [AppendResult(int(r["seq"]), bool(r.get("deduped"))) for r in out["results"]]

    async def info(self) -> StreamInfo:
        """The stream's latest sequence number, whether it is sealed, and when it was created, last written
        and is due to expire."""
        async with self._lock:
            await self._create_locked()
        data, _ = await self._call("GET", self._path)
        run = json.loads(data)
        expires_us = run.get("expires_ts_us")
        return StreamInfo(
            name=self.name,
            last_seq=int(run["head_seq"]),
            sealed=bool(run["sealed"]),
            created_at=_utc(run["created_ts_us"]),
            last_written_at=_utc(run["last_event_ts_us"]),
            expires_at=None if expires_us is None else _utc(expires_us),
        )

    async def __getitem__(self, seq: int) -> Entry:
        """The entry at sequence number `seq`; raises `IndexError` past the end of the stream. Use `read` for
        a range of entries."""
        if not isinstance(seq, int):
            raise TypeError("Streams are indexed by sequence number; use `read` for a range of entries.")
        if seq < 1:
            raise IndexError(f"Stream '{self.name}' sequence numbers start at 1.")
        async with self._lock:
            await self._create_locked()
        data, _ = await self._call("GET", f"{self._path}/events", params={"from_seq": seq, "limit": 1})
        line = next((line for line in data.splitlines() if line.strip()), None)
        if line is None:
            raise IndexError(f"Stream '{self.name}' has no entry {seq}.")
        return _entry(json.loads(line))

    async def read(self, start: int = 1, *, follow: bool = False) -> AsyncGenerator[Entry, None]:
        """Iterate over the stream's entries from sequence number `start`.

        With `follow=True`, keep waiting for new entries until the stream is sealed, reconnecting after the
        last entry when the connection drops.
        """
        async with self._lock:
            await self._create_locked()
        entries = self._follow(start) if follow else self._replay(start)
        async for entry in entries:
            yield entry

    async def _replay(self, start: int) -> AsyncGenerator[Entry, None]:
        while True:
            data, headers = await self._call(
                "GET", f"{self._path}/events", params={"from_seq": start, "limit": READ_PAGE_SIZE}
            )
            events = [json.loads(line) for line in data.splitlines() if line.strip()]
            for event in events:
                yield _entry(event)
            start = int(headers["x-dlog-next-seq"])
            if not events or start > int(headers["x-dlog-head-seq"]):
                return

    async def _follow(self, start: int) -> AsyncGenerator[Entry, None]:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=REQUEST_TIMEOUT_S, sock_read=FOLLOW_IDLE_TIMEOUT_S)
        failures, deadline = 0, 0.0
        while True:
            params = {"follow": "true", "from_seq": start}
            try:
                async with ClientSessionRegistry.get_session().get(
                    _service_url() + self._path + "/events",
                    params=params,
                    headers=await self._headers(),
                    timeout=timeout,
                ) as resp:
                    if resp.status >= 400:
                        raise _error_for(resp.status, await resp.read(), self.name)
                    async for event, data in _sse_messages(resp.content):
                        if event == "sealed":
                            return
                        if event == "error":
                            err = json.loads(data)
                            raise _error_for(
                                int(err.get("status", 500)), json.dumps({"error": err}).encode(), self.name
                            )
                        entry = _entry(json.loads(data))
                        start, failures = entry.seq + 1, 0
                        yield entry
            except (aiohttp.ClientError, asyncio.TimeoutError, _Transient) as exc:
                failures += 1
                if failures == 1:
                    deadline = time.monotonic() + RETRY_DEADLINE_S
                delay = _backoff(failures)
                if time.monotonic() + delay > deadline:
                    raise self._unreachable(exc) from exc
                await asyncio.sleep(delay)
                continue
            # The service ended the follow without a seal, as it does every few minutes; reopen it after the last entry.
            failures = 0
            await asyncio.sleep(FOLLOW_REOPEN_DELAY_S)

    async def fork(self, name: str, *, at: int | None = None, retention: timedelta | None = None) -> "_Stream":
        """Create a Stream named `name` that starts with this stream's entries through sequence number `at`,
        or all of them but a final `seal` by default, and continues independently from there.

        The fork reads those entries from this stream without copying them, and keeps them even if this
        stream is deleted or expires. `retention` applies to the fork as in `Stream.from_name`; without
        it the fork keeps this stream's retention.
        """
        _check_name(name)
        _check_retention(retention)
        async with self._lock:
            await self._create_locked()
        if at is None:
            # A fixed fork point lets a retry recognize the fork its first attempt created.
            data, _ = await self._call("GET", self._path)
            parent = json.loads(data)
            at = int(parent["sealed_seq"]) - 1 if parent.get("sealed") else int(parent["head_seq"])
        body: dict[str, Any] = {"run_id": name, "parent_run_id": self.name, "fork_seq": at}
        if retention is not None:
            body["retention_s"] = int(retention.total_seconds())
        fork = _Stream(name, self._environment_name, False, None, self._client)
        try:
            data, _ = await self._call("POST", "/v1/runs", body)
        except AlreadyExistsError:
            # A retried create whose first attempt succeeded finds the fork it made.
            data, _ = await fork._call("GET", fork._path)
            existing = json.loads(data)
            if existing.get("parent_run_id") != self.name or existing.get("fork_seq") != at:
                raise
        fork._gen = json.loads(data)["gen"]
        return fork

    async def seal(self) -> int:
        """Close the stream to writers and return the sequence number of its final `seal` entry."""
        return int((await self._write("/seal", {}))["sealed_seq"])

    def writer(
        self,
        *,
        on_written: Callable[[str, AppendResult], None] | None = None,
        on_conflict: Callable[[IdempotencyConflict], None] | None = None,
    ) -> "_StreamWriter":
        """A writer that appends in the background, batching values that arrive while a write is in flight."""
        return _StreamWriter(self, on_written, on_conflict)


class _StreamWriter:
    """Appends to a Stream from a background task, in order and in batches.

    `append` queues a value and returns at once, or waits while 1000 values are queued. Do not change a
    value after appending it. Each value written is reported to `on_written` with its idempotency key and
    result, and a reused key with a different value to `on_conflict`. A write that fails after retries, or
    a conflict without an `on_conflict` handler, stops the writer: the error is raised from the next
    `append`, `flush` or `close`, and values still queued are dropped. Use the writer as an async context
    manager or call `close`, so every value is written or the error is raised. Callbacks run on Modal's
    event loop thread, not the caller's, and should return quickly.
    """

    def __init__(
        self,
        stream: _Stream,
        on_written: Callable[[str, AppendResult], None] | None,
        on_conflict: Callable[[IdempotencyConflict], None] | None,
    ):
        """mdmd:hidden"""
        self._stream = stream
        self._on_written = on_written
        self._on_conflict = on_conflict
        self._queue: list[tuple[str, Any]] = []
        self._error: BaseException | None = None
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._progress = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._closed = False

    async def append(self, value: Any, *, idempotency_key: str | None = None) -> None:
        """Queue a value to be appended; see `Stream.append` for the limits and idempotency."""
        _check_value(value)
        while not self._try_enqueue(value, idempotency_key):
            self._progress.clear()
            await self._progress.wait()
        self._start()

    def _try_enqueue(self, value: Any, idempotency_key: str | None) -> bool:
        """Queue the value and return True, or return False while the queue is full."""
        if self._closed:
            raise InvalidError("This StreamWriter is closed.")
        if self._error is not None:
            raise self._error
        if len(self._queue) >= WRITER_MAX_PENDING:
            return False
        self._queue.append((uuid.uuid4().hex if idempotency_key is None else idempotency_key, value))
        return True

    def _start(self) -> None:
        if self._closed and not self._queue:
            return
        self._idle.clear()
        self._wake.set()
        # A `close` cancelled mid-write leaves a finished task behind; a new one resumes the queue.
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def flush(self) -> None:
        """Wait until every queued value is written."""
        while True:
            await self._idle.wait()
            if self._error is not None:
                raise self._error
            if not self._queue:
                return
            # A cancelled `close` stops the writer with values still queued.
            self._start()

    async def close(self) -> None:
        """Flush and stop the background task; an `append` still waiting for room raises `InvalidError`."""
        self._closed = True
        self._progress.set()
        try:
            await self.flush()
        finally:
            if self._task is not None:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)

    async def __aenter__(self) -> "_StreamWriter":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _run(self) -> None:
        try:
            await self._drain()
        finally:
            self._idle.set()

    async def _drain(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self._queue:
                batch, keys = [], set()
                for key, value in self._queue[:WRITER_MAX_BATCH]:
                    if key in keys:
                        break
                    batch.append((key, value))
                    keys.add(key)
                try:
                    await self._write(batch)
                except Exception as exc:
                    logger.warning(f"Stream '{self._stream.name}' writer stopped: {exc}")
                    self._error = exc
                    self._queue.clear()
                del self._queue[: len(batch)]
                self._progress.set()
            self._idle.set()

    async def _write(self, batch: list[tuple[str, Any]]) -> None:
        while batch:
            try:
                results = await self._stream.append_many([v for _, v in batch], idempotency_keys=[k for k, _ in batch])
            except IdempotencyConflict as exc:
                rest = [item for item in batch if item[0] != exc.idempotency_key]
                if self._on_conflict is None or len(rest) == len(batch):
                    raise
                self._report(self._on_conflict, exc)
                # A conflict writes nothing, so the rest of the batch is resent.
                batch = rest
                continue
            if self._on_written is not None:
                for (key, _), result in zip(batch, results):
                    self._report(self._on_written, key, result)
            return

    def _report(self, callback: Callable[..., None], *args: Any) -> None:
        try:
            callback(*args)
        except Exception:
            logger.exception(f"Stream '{self._stream.name}' writer callback raised")


Stream = synchronize_api(_Stream, target_module=__name__)
StreamWriter = synchronize_api(_StreamWriter, target_module=__name__)
