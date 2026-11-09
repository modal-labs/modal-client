# Copyright Modal Labs 2026
import asyncio
import json
import os
import pytest
import time
from datetime import datetime, timedelta, timezone

import aiohttp.web
import jwt
import pytest_asyncio

from modal._utils.http_utils import run_temporary_http_server
from modal.client import Client
from modal.exception import AlreadyExistsError, InvalidError, NotFoundError, ServiceError
from modal.experimental import IdempotencyConflict, Stream, StreamInfo
from modal.experimental.stream import AppendResult
from modal_proto import api_pb2

WORKSPACE_ID = "ac-" + "a" * 22


def _error(status: int, code: str, **extra):
    return aiohttp.web.json_response({"error": {"code": code, "message": code, **extra}}, status=status)


class FakeStreamsService:
    """The streams HTTP API for one environment, stamping entry `seq` at `seq` seconds past the epoch."""

    def __init__(self):
        self.runs: dict[str, dict] = {}
        self.requests: list[aiohttp.web.Request] = []
        # Ends the next follow response, without a seal, after this many entries.
        self.drop_follow_after: int | None = None
        # Per follow request in order: an error status to send, or 0 to end the response with no entries.
        self.follow_script: list[int] = []
        # Creates and deletes that apply and then answer 503, as if the response were lost.
        self.lose_write_responses = 0
        self.changed = asyncio.Condition()
        self.app = aiohttp.web.Application(middlewares=[self._record])
        self.app.add_routes(
            [
                aiohttp.web.post("/v1/runs", self.create),
                aiohttp.web.get("/v1/runs/{run}", self.get_run),
                aiohttp.web.delete("/v1/runs/{run}", self.delete),
                aiohttp.web.post("/v1/runs/{run}/append", self.append),
                aiohttp.web.post("/v1/runs/{run}/seal", self.seal),
                aiohttp.web.get("/v1/runs/{run}/events", self.events),
            ]
        )

    @aiohttp.web.middleware
    async def _record(self, request, handler):
        self.requests.append(request)
        return await handler(request)

    async def create(self, request):
        body = await request.json()
        if body["run_id"] in self.runs:
            return _error(409, "already_exists")
        events, retention_s = [], body.get("retention_s")
        if (parent := body.get("parent_run_id")) is not None:
            if parent not in self.runs:
                return _error(404, "not_found")
            parent_events = [e for e in self.runs[parent]["events"] if e["kind"] != "seal"]
            events = parent_events[: body.get("fork_seq", len(parent_events))]
            retention_s = retention_s or self.runs[parent]["retention_s"]
        self.runs[body["run_id"]] = {
            "gen": 1,
            "events": events,
            "sealed": False,
            "retention_s": retention_s,
            "parent_run_id": parent,
            "fork_seq": len(events) if parent else None,
        }
        return self._maybe_lose(aiohttp.web.json_response({"run_id": body["run_id"], "gen": 1}))

    def _maybe_lose(self, response):
        if self.lose_write_responses:
            self.lose_write_responses -= 1
            return _error(503, "unavailable")
        return response

    async def delete(self, request):
        if self.runs.pop(request.match_info["run"], None) is None:
            return _error(404, "not_found")
        return self._maybe_lose(aiohttp.web.Response(status=204))

    async def get_run(self, request):
        if (run := self.runs.get(request.match_info["run"])) is None:
            return _error(404, "not_found")
        fields = {
            "gen": run["gen"],
            "head_seq": len(run["events"]),
            "sealed": run["sealed"],
            "created_ts_us": 0,
            "last_event_ts_us": len(run["events"]) * 1_000_000,
            "expires_ts_us": run["retention_s"] and (len(run["events"]) + run["retention_s"]) * 1_000_000,
        }
        if run["sealed"]:
            fields["sealed_seq"] = len(run["events"])
        return aiohttp.web.json_response(
            {**fields, "parent_run_id": run.get("parent_run_id"), "fork_seq": run.get("fork_seq")}
        )

    async def append(self, request):
        if (run := self.runs.get(request.match_info["run"])) is None:
            return _error(404, "not_found")
        body = await request.json()
        stored = {e["op_id"]: e for e in run["events"]}
        for ev in body["events"]:
            if (prior := stored.get(ev["op_id"])) is not None and prior["value"] != ev["value"]:
                return _error(409, "idempotency_conflict", op_id=ev["op_id"], seq=prior["seq"])
        if any(ev["op_id"] not in stored for ev in body["events"]) and body["gen"] != run["gen"]:
            return _error(409, "fenced")
        results = []
        for ev in body["events"]:
            if (prior := stored.get(ev["op_id"])) is None:
                prior = stored[ev["op_id"]] = {**ev, "seq": len(run["events"]) + 1}
                prior["ts_us"] = prior["seq"] * 1_000_000
                run["events"].append(prior)
                results.append({"seq": prior["seq"], "deduped": False})
            else:
                results.append({"seq": prior["seq"], "deduped": True})
        async with self.changed:
            self.changed.notify_all()
        return aiohttp.web.json_response({"results": results})

    async def seal(self, request):
        run = self.runs[request.match_info["run"]]
        if not run["sealed"]:
            seq = len(run["events"]) + 1
            seal = {"op_id": "dlog:seal", "kind": "seal", "value": {"gen": run["gen"]}}
            run["events"].append({**seal, "seq": seq, "ts_us": seq * 1_000_000})
            run["sealed"] = True
        async with self.changed:
            self.changed.notify_all()
        return aiohttp.web.json_response({"sealed_seq": len(run["events"])})

    async def events(self, request):
        if (run := self.runs.get(request.match_info["run"])) is None:
            return _error(404, "not_found")
        q, events = request.query, run["events"]
        start = int(q.get("from_seq", 1))
        if q.get("follow") != "true":
            page = events[start - 1 : start - 1 + int(q["limit"])]
            headers = {"x-dlog-head-seq": str(len(events)), "x-dlog-next-seq": str(start + len(page))}
            return aiohttp.web.Response(text="".join(json.dumps(e) + "\n" for e in page), headers=headers)
        resp = aiohttp.web.StreamResponse(headers={"content-type": "text/event-stream"})
        await resp.prepare(request)
        if self.follow_script:
            if status := self.follow_script.pop(0):
                error = {"status": status, "code": "unavailable", "message": "read failed"}
                await resp.write(f"event: error\ndata: {json.dumps(error)}\n\n".encode())
            return resp
        drop_after, self.drop_follow_after = self.drop_follow_after, None
        sent = 0
        while True:
            while start <= len(events):
                if sent == drop_after:
                    return resp
                await resp.write(f"id: {start}\ndata: {json.dumps(events[start - 1])}\n\n".encode())
                start, sent = start + 1, sent + 1
            if run["sealed"]:
                await resp.write(b"event: sealed\ndata:\n\n")
                return resp
            async with self.changed:
                await self.changed.wait()


@pytest_asyncio.fixture
async def service(monkeypatch, servicer):
    servicer.auth_token = jwt.encode(
        {"exp": int(time.time()) + 3600, "workspace_id": WORKSPACE_ID},
        "a-shared-secret-of-at-least-32-bytes",
        algorithm="HS256",
    )
    fake = FakeStreamsService()
    async with run_temporary_http_server(fake.app) as url:
        monkeypatch.setenv("MODAL_STREAMS_URL", url)
        yield fake


@pytest.mark.asyncio
async def test_from_name_and_append(client, servicer, service, monkeypatch):
    with pytest.raises(NotFoundError):
        await Stream.from_name("missing", client=client).append.aio(1)

    stream = Stream.from_name("s", create_if_missing=True, retention=timedelta(days=2), client=client)
    assert await stream.append.aio("a") == AppendResult(seq=1, deduped=False)
    assert service.runs["s"]["retention_s"] == 2 * 86400
    with pytest.raises(InvalidError, match="at least one hour"):
        Stream.from_name("s", create_if_missing=True, retention=timedelta(minutes=59))
    # A second handle finds the stream and writes alongside the first.
    other = Stream.from_name("s", create_if_missing=True, client=client)
    assert await other.append.aio("b") == AppendResult(seq=2, deduped=False)
    assert await stream.append.aio("c") == AppendResult(seq=3, deduped=False)
    # A writer outside this API takes a new generation; the handle picks it up and resends.
    service.runs["s"]["gen"] = 2
    assert await stream.append.aio("d") == AppendResult(seq=4, deduped=False)

    assert {r.headers["x-streams-environment"] for r in service.requests} == {"main"}
    assert {r.headers["Authorization"] for r in service.requests} == {f"Bearer {servicer.auth_token}"}

    with pytest.raises(InvalidError, match="JSON"):
        await stream.append.aio(float("nan"))
    with pytest.raises(InvalidError, match="65536 bytes"):
        await stream.append.aio("x" * 65536)
    with pytest.raises(InvalidError, match="surrogates"):
        await stream.append.aio("\ud800")
    with pytest.raises(InvalidError, match="Invalid Stream name"):
        Stream.from_name("a.b", client=client)

    # An empty setting means the default service, here pointed at the fake one.
    monkeypatch.setattr("modal.experimental.stream.DEFAULT_STREAMS_URL", os.environ["MODAL_STREAMS_URL"])
    monkeypatch.setenv("MODAL_STREAMS_URL", "")
    assert await Stream.from_name("s", client=client).append.aio("e") == AppendResult(seq=5, deduped=False)

    sent = len(service.requests)
    for url in ("https://streams.example.com", "https://localhost.example.com", "https://127.0.0.1.nip.io"):
        monkeypatch.setenv("MODAL_STREAMS_URL", url)
        with pytest.raises(InvalidError, match="localhost"):
            await Stream.from_name("s", client=client).append.aio("e")
    assert len(service.requests) == sent


@pytest.mark.asyncio
async def test_default_environment_is_per_client(client, servicer, service, credentials):
    await Stream.from_name("s", create_if_missing=True, client=client).append.aio(1)
    # A second client whose workspace has a different default environment.
    servicer.environments = {"staging": "en-2", **servicer.environments}
    async with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as other:
        await Stream.from_name("s", create_if_missing=True, client=other).append.aio(2)
    assert [r.headers["x-streams-environment"] for r in service.requests] == ["main"] * 2 + ["staging"] * 3


@pytest.mark.asyncio
async def test_name_limits(client, service):
    await Stream.from_name("n" * 64, create_if_missing=True, client=client).append.aio(1)
    for name in ["n" * 65, "", "a.b"]:
        with pytest.raises(InvalidError, match="1-64 alphanumeric"):
            Stream.from_name(name)


@pytest.mark.asyncio
async def test_append_many_dedupe(client, service):
    stream = Stream.from_name("s", create_if_missing=True, client=client)
    assert await stream.append.aio({"n": 1}, idempotency_key="k1") == AppendResult(1, False)
    assert await stream.append.aio({"n": 1}, idempotency_key="k1") == AppendResult(1, True)
    results = await stream.append_many.aio([{"n": 1}, {"n": 2}], idempotency_keys=["k1", "k2"])
    assert results == [AppendResult(1, True), AppendResult(2, False)]

    with pytest.raises(IdempotencyConflict) as exc_info:
        await stream.append_many.aio([{"n": 3}, {"n": 99}], idempotency_keys=["k3", "k2"])
    assert (exc_info.value.idempotency_key, exc_info.value.seq) == ("k2", 2)
    assert len(service.runs["s"]["events"]) == 2
    with pytest.raises(InvalidError, match="distinct"):
        await stream.append_many.aio([{"n": 3}, {"n": 4}], idempotency_keys=["k3", "k3"])
    assert await stream.append_many.aio(v for v in [{"n": 5}]) == [AppendResult(3, False)]


@pytest.mark.asyncio
async def test_fork(client, service, monkeypatch):
    monkeypatch.setattr("modal.experimental.stream.RETRY_MAX_DELAY_S", 0)
    stream = Stream.from_name("s", create_if_missing=True, retention=timedelta(days=3), client=client)
    await stream.append_many.aio(["a", "b", "c"])
    fork = await stream.fork.aio("f", at=2, retention=timedelta(hours=2))
    assert service.runs["f"]["retention_s"] == 7200
    assert await fork.append.aio("x") == AppendResult(seq=3, deduped=False)
    assert [e.value async for e in fork.read.aio()] == ["a", "b", "x"]
    assert [e.value async for e in stream.read.aio()] == ["a", "b", "c"]

    # A fork created by a request whose response was lost is found by the retry.
    service.lose_write_responses = 1
    whole = await stream.fork.aio("g")
    assert service.runs["g"]["retention_s"] == 3 * 86400
    assert [e.value async for e in whole.read.aio()] == ["a", "b", "c"]
    with pytest.raises(AlreadyExistsError):
        await stream.fork.aio("f", at=1)
    with pytest.raises(AlreadyExistsError):
        await whole.fork.aio("f")
    # An older fork of the same parent is not mistaken for one of all its current entries.
    await stream.append.aio("d")
    with pytest.raises(AlreadyExistsError):
        await stream.fork.aio("g")


@pytest.mark.asyncio
async def test_delete(client, service):
    stream = Stream.from_name("s", create_if_missing=True, client=client)
    await stream.append.aio("a")
    await Stream.objects.delete.aio("s", client=client)
    with pytest.raises(NotFoundError):
        await stream.append.aio("b")
    with pytest.raises(NotFoundError):
        await Stream.objects.delete.aio("s", client=client)
    await Stream.objects.delete.aio("s", allow_missing=True, client=client)

    # A delete is sent once, so a lost response surfaces instead of being resent.
    await Stream.from_name("t", create_if_missing=True, client=client).append.aio("a")
    service.lose_write_responses = 1
    with pytest.raises(ServiceError):
        await Stream.objects.delete.aio("t", client=client)
    assert [r.method for r in service.requests].count("DELETE") == 4


@pytest.mark.asyncio
async def test_read_and_info(client, service):
    stream = Stream.from_name("s", create_if_missing=True, client=client)
    await stream.append_many.aio(["a", "b", "c"], idempotency_keys=["k1", "k2", "k3"])

    entries = [e async for e in stream.read.aio()]
    assert [(e.seq, e.idempotency_key, e.kind, e.value) for e in entries] == [
        (1, "k1", "value", "a"),
        (2, "k2", "value", "b"),
        (3, "k3", "value", "c"),
    ]
    assert entries[1].ts == datetime.fromtimestamp(2, tz=timezone.utc)
    assert [e.value async for e in stream.read.aio(3)] == ["c"]
    assert (await asyncio.to_thread(lambda: stream[2])).value == "b"
    for seq in (4, 0):
        with pytest.raises(IndexError):
            await asyncio.to_thread(stream.__getitem__, seq)
    with pytest.raises(TypeError):
        await asyncio.to_thread(stream.__getitem__, slice(1, 2))  # type: ignore[arg-type]

    epoch = datetime.fromtimestamp(0, tz=timezone.utc)
    assert await stream.info.aio() == StreamInfo("s", 3, False, epoch, epoch + timedelta(seconds=3), None)
    kept = Stream.from_name("kept", create_if_missing=True, retention=timedelta(hours=1), client=client)
    info = await kept.info.aio()
    assert (info.last_seq, info.expires_at) == (0, epoch + timedelta(hours=1))


@pytest.mark.asyncio
async def test_follow_resumes_after_dropped_connection(client, service):
    stream = Stream.from_name("s", create_if_missing=True, client=client)
    await stream.append_many.aio(["a", "b", "c"])
    service.drop_follow_after = 1

    seen = []
    async for entry in stream.read.aio(2, follow=True):
        seen.append(entry.value)
        if entry.value == "c":
            await stream.append.aio("d")
            await stream.seal.aio()
    assert seen == ["b", "c", "d", {"gen": 1}]
    follows = [dict(r.query) for r in service.requests if r.query.get("follow") == "true"]
    assert follows == [{"follow": "true", "from_seq": "2"}, {"follow": "true", "from_seq": "3"}]


@pytest.mark.asyncio
async def test_follow_gives_up_on_repeated_errors(client, service, monkeypatch):
    # Each scripted error waits 0.02 s, far inside the 0.5 s deadline, so scheduling delays cannot flake this.
    for name, value in [("RETRY_DEADLINE_S", 0.5), ("RETRY_BASE_DELAY_S", 0.02), ("RETRY_MAX_DELAY_S", 0.02)]:
        monkeypatch.setattr(f"modal.experimental.stream.{name}", value)
    stream = Stream.from_name("s", create_if_missing=True, client=client)
    service.follow_script = [503] * 100
    with pytest.raises(ServiceError, match="read failed"):
        async for _ in stream.read.aio(follow=True):
            pass
    assert 1 < 100 - len(service.follow_script) < 100
    # Errors between connections that end cleanly do not add up, though together they outlast the deadline.
    service.follow_script = [503, 503, 0] * 15
    await stream.seal.aio()
    assert [e.kind async for e in stream.read.aio(follow=True)] == ["seal"]
    service.follow_script = [410]
    with pytest.raises(NotFoundError):
        async for _ in stream.read.aio(follow=True):
            pass
