# Copyright Modal Labs 2026
import asyncio
import pytest
from datetime import datetime, timedelta, timezone

from grpclib import GRPCError, Status

import modal
from modal.exception import InvalidError
from modal.types import ObjectMetrics
from modal_proto import api_pb2

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _handle(kind, client):
    return (modal.Function if kind == "function" else modal.Server).from_id("fu-test", client=client)


def _get_response(kind):
    return api_pb2.FunctionGetByIdResponse(
        function=api_pb2.FunctionData(
            is_server=kind == "server",
            ranked_functions=[api_pb2.FunctionData.RankedFunction(function=api_pb2.Function())],
        )
    )


def _samples(kind):
    response = getattr(api_pb2, "FunctionGetMetricsResponse" if kind == "function" else "ServerGetMetricsResponse")(
        bucket_secs=60
    )
    response.since.FromDatetime(START)
    response.until.FromDatetime(START + timedelta(minutes=2))
    series = response.series.add(name="future_usage_avg", unit="widgets", description="A future metric.")
    series.points.add(value=0).timestamp.FromDatetime(START)
    series.points.add().timestamp.FromDatetime(START + timedelta(minutes=1))
    return response


@pytest.mark.parametrize("kind", ["function", "server"])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_metrics_export(client, servicer, kind, async_call):
    obj = _handle(kind, client)
    rpc = "FunctionGetMetrics" if kind == "function" else "ServerGetMetrics"
    kwargs = dict(
        since=START + timedelta(seconds=15),
        until=START + timedelta(minutes=2, seconds=15),
        bucket_size=timedelta(minutes=1),
        groups=["future"],
    )
    if kind == "function":
        kwargs["all_variants"] = True
    with servicer.intercept() as ctx:
        ctx.add_response("FunctionGetById", _get_response(kind))
        ctx.add_response(rpc, _samples(kind))
        result = (
            await obj.metrics.export.aio(**kwargs)
            if async_call
            else await asyncio.to_thread(obj.metrics.export, **kwargs)
        )
    assert isinstance(result, ObjectMetrics)
    assert result.object_id == "fu-test"
    assert result.since == START and result.until == START + timedelta(minutes=2)
    assert result.bucket_size == timedelta(minutes=1)
    assert result.point_count == 2
    assert result["future_usage_avg"].aggregation == "avg"
    assert result["future_usage_avg"].description == "A future metric."
    assert [point.value for point in result.series["future_usage_avg"].points] == [0, None]
    assert result.series["future_usage_avg"].points[0].time == START
    with pytest.raises(KeyError):
        result["missing"]
    ctx.pop_request("FunctionGetById")
    request = ctx.pop_request(rpc)
    assert request.function_id == "fu-test"
    assert request.since.ToDatetime(tzinfo=timezone.utc) == kwargs["since"]
    assert request.bucket_secs == 60
    assert list(request.groups) == ["future"]
    if kind == "function":
        assert request.rollup
    assert not ctx.calls


@pytest.mark.parametrize("kind", ["function", "server"])
def test_metrics_defaults(client, servicer, kind):
    rpc = "FunctionGetMetrics" if kind == "function" else "ServerGetMetrics"
    before = datetime.now(timezone.utc)
    with servicer.intercept() as ctx:
        ctx.add_response("FunctionGetById", _get_response(kind))
        ctx.add_response(rpc, _samples(kind))
        _handle(kind, client).metrics.export()
    request = ctx.pop_request(rpc)
    until = request.until.ToDatetime(tzinfo=timezone.utc)
    assert before <= until <= datetime.now(timezone.utc)
    assert until - request.since.ToDatetime(tzinfo=timezone.utc) == timedelta(hours=1)
    assert not request.HasField("bucket_secs") and not request.groups
    if kind == "function":
        assert not request.rollup


@pytest.mark.parametrize("kind", ["function", "server"])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_metrics_schema(client, servicer, kind, async_call):
    schema = api_pb2.MetricsGetInfoResponse(bucket_secs=60, bucket_count=60)
    for label in ("future_{a,b}_avg", "future_c_count"):
        row = schema.groups.add(name="future", display_name=label, description="Server description.")
        row.metrics.add(name=label, unit="widgets", description="Series description.")
    with servicer.intercept() as ctx:
        ctx.add_response("FunctionGetById", _get_response(kind))
        ctx.add_response("MetricsGetInfo", schema)
        manager = _handle(kind, client).metrics
        result = (
            await manager.schema.aio(["future"]) if async_call else await asyncio.to_thread(manager.schema, ["future"])
        )
    assert list(result) == ["future"]
    assert result["future"].name == "future"
    assert [m.name for m in result["future"].definitions] == ["future_{a,b}_avg", "future_c_count"]
    assert result["future"].definitions[0].description == "Series description."
    assert result["future"].definitions[0].unit == "widgets"
    ctx.pop_request("FunctionGetById")
    request = ctx.pop_request("MetricsGetInfo")
    assert list(request.groups) == ["future"]
    target = api_pb2.MetricsGetInfoRequest
    assert request.target_type == (
        target.METRICS_TARGET_TYPE_FUNCTION if kind == "function" else target.METRICS_TARGET_TYPE_SERVER
    )
    assert not ctx.calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"since": START, "until": START},
        {"since": START, "until": START + timedelta(days=32)},
        {"bucket_size": timedelta(seconds=59)},
        {"bucket_size": timedelta(seconds=60.5)},
        {"since": START, "until": START + timedelta(days=1), "bucket_size": timedelta(minutes=1)},
        {"groups": "calls"},
    ],
)
def test_metrics_invalid_arguments(client, servicer, kwargs):
    with servicer.intercept() as ctx:
        with pytest.raises(InvalidError):
            _handle("function", client).metrics.export(**kwargs)
    assert not ctx.calls


def test_metrics_naive_datetimes_and_until_default():
    from modal._metrics import _metrics_request

    until = datetime(2026, 10, 1)
    request = _metrics_request(None, until, None, None)
    assert request.until.ToDatetime(tzinfo=timezone.utc) == until.astimezone(timezone.utc)
    assert request.until.seconds - request.since.seconds == 3600


def test_metrics_schema_rejects_unknown_group(client, servicer):
    async def reject_groups(servicer, stream):
        request = await stream.recv_message()
        assert list(request.groups) == ["unknown"]
        raise GRPCError(Status.INVALID_ARGUMENT, "Invalid metrics groups: unknown")

    with servicer.intercept() as ctx:
        ctx.add_response("FunctionGetById", _get_response("function"))
        ctx.set_responder("MetricsGetInfo", reject_groups)
        with pytest.raises(GRPCError, match="Invalid metrics groups: unknown"):
            _handle("function", client).metrics.schema(["unknown"])


def test_metrics_empty_export(client, servicer):
    response = api_pb2.FunctionGetMetricsResponse(bucket_secs=60)
    response.since.FromDatetime(START)
    response.until.FromDatetime(START)
    with servicer.intercept() as ctx:
        ctx.add_response("FunctionGetById", _get_response("function"))
        ctx.add_response("FunctionGetMetrics", response)
        result = _handle("function", client).metrics.export(since=START, until=START + timedelta(seconds=30))
    assert result.series == {} and result.point_count == 0
    assert result.since == result.until == START
