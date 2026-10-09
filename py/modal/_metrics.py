# Copyright Modal Labs 2026
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from modal_proto import api_pb2

from ._object import _Object
from ._utils.async_utils import synchronize_api
from .exception import InvalidError
from .types import MetricDefinition, MetricSeries, MetricsSchema, ObjectMetrics, TimeSeriesPoint


def _metrics_request(
    since: datetime | None, until: datetime | None, bucket_size: timedelta | None, groups: list[str] | None
) -> api_pb2.FunctionGetMetricsRequest:
    until = (until or datetime.now(timezone.utc)).astimezone(timezone.utc)
    since = (since or until - timedelta(hours=1)).astimezone(timezone.utc)
    if since >= until:
        raise InvalidError("`since` must be before `until`.")
    if since.timestamp() < 0:
        raise InvalidError("`since` must be at or after the Unix epoch.")
    if until - since > timedelta(days=31):
        raise InvalidError("Metrics time range cannot exceed 31 days.")
    if groups is not None and (not isinstance(groups, list) or any(not isinstance(g, str) or not g for g in groups)):
        raise InvalidError("`groups` must be a list of nonempty group names.")
    request = api_pb2.FunctionGetMetricsRequest(groups=groups or [])
    request.since.FromDatetime(since)
    request.until.FromDatetime(until)
    if bucket_size is not None:
        if not isinstance(bucket_size, timedelta):
            raise InvalidError("`bucket_size` must be a timedelta.")
        seconds = bucket_size.total_seconds()
        if not seconds.is_integer() or not 60 <= seconds <= 2**32 - 1:
            raise InvalidError("`bucket_size` must be a whole number of seconds, at least one minute.")
        request.bucket_secs = int(seconds)
        if int(until.timestamp()) // request.bucket_secs - int(since.timestamp()) // request.bucket_secs > 500:
            raise InvalidError("Metrics are limited to 500 points per series; increase `bucket_size`.")
    return request


def _object_metrics(
    object_id: str, response: api_pb2.FunctionGetMetricsResponse | api_pb2.ServerGetMetricsResponse
) -> ObjectMetrics:
    series = {
        item.name: MetricSeries(
            name=item.name,
            unit=item.unit,
            aggregation=item.name.rsplit("_", 1)[-1],
            description=item.description,
            points=[
                TimeSeriesPoint(
                    time=point.timestamp.ToDatetime(tzinfo=timezone.utc),
                    value=point.value if point.HasField("value") else None,
                )
                for point in item.points
            ],
        )
        for item in response.series
    }
    return ObjectMetrics(
        object_id=object_id,
        since=response.since.ToDatetime(tzinfo=timezone.utc),
        until=response.until.ToDatetime(tzinfo=timezone.utc),
        bucket_size=timedelta(seconds=response.bucket_secs),
        point_count=sum(len(item.points) for item in series.values()),
        series=series,
    )


class _MetricsManager:
    """mdmd:namespace"""

    def __init__(self, function: _Object, *, is_server: bool):
        """mdmd:hidden"""
        self._function = function
        self._is_server = is_server

    async def schema(self, groups: list[str] | None = None) -> dict[str, MetricsSchema]:
        """Return metric definitions keyed by group name without fetching samples.

        Omit `groups` to list all groups. Each group contains its exact individual
        series definitions.
        """
        response = await self._schema(groups=groups)
        schema: dict[str, MetricsSchema] = {}
        for row in response.groups:
            if row.name not in schema:
                schema[row.name] = MetricsSchema(name=row.name, definitions=[])
            schema[row.name].definitions.extend(
                MetricDefinition(name=m.name, unit=m.unit, description=m.description) for m in row.metrics
            )
        return schema

    async def _schema(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        bucket_size: timedelta | None = None,
        groups: list[str] | None = None,
    ) -> api_pb2.MetricsGetInfoResponse:
        """Fetch schema rows and resolved bucket metadata for a preview window."""
        window = _metrics_request(since, until, bucket_size, groups)
        await self._function.hydrate()
        target = api_pb2.MetricsGetInfoRequest
        request = target(
            target_type=target.METRICS_TARGET_TYPE_SERVER if self._is_server else target.METRICS_TARGET_TYPE_FUNCTION,
            since=window.since,
            until=window.until,
            groups=groups,
        )
        if window.HasField("bucket_secs"):
            request.bucket_secs = window.bucket_secs
        return await self._function.client._stub.MetricsGetInfo(request)

    async def _export(
        self,
        since: datetime | None,
        until: datetime | None,
        bucket_size: timedelta | None,
        groups: list[str] | None,
        all_variants: bool = False,
    ) -> ObjectMetrics:
        request = _metrics_request(since, until, bucket_size, groups)
        await self._function.hydrate()
        request.function_id = self._function.object_id
        if self._is_server:
            server_request = api_pb2.ServerGetMetricsRequest(
                function_id=request.function_id,
                since=request.since,
                until=request.until,
                groups=request.groups,
            )
            if request.HasField("bucket_secs"):
                server_request.bucket_secs = request.bucket_secs
            response = await self._function.client._stub.ServerGetMetrics(server_request)
        else:
            request.rollup = all_variants
            response = await self._function.client._stub.FunctionGetMetrics(request)
        return _object_metrics(self._function.object_id, response)


class _FunctionMetricsManager(_MetricsManager):
    """mdmd:namespace"""

    def __init__(self, function: _Object):
        """mdmd:hidden"""
        super().__init__(function, is_server=False)

    async def export(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        bucket_size: timedelta | None = None,
        groups: list[str] | None = None,
        all_variants: bool = False,
    ) -> ObjectMetrics:
        """Fetch metric samples for a Function.

        Defaults to all groups over the last hour. `since` defaults to one hour
        before `until`; naive datetimes use local time. The maximum range is 31
        days, with at most 500 points per series. If `bucket_size` is omitted,
        Modal selects a supported width targeting approximately 100 buckets.
        Bounds round down to UTC bucket boundaries, so data before `since` may
        be included and the partial bucket at `until` is excluded.

        Set `all_variants=True` to include the root Function and its variants.
        Missing observations have value None. Use `schema()` to discover
        group names and series definitions without fetching samples.

        ```python notest
        metrics = function.metrics.export(bucket_size=timedelta(minutes=5), groups=["calls"])
        for point in metrics["successful_calls_count"].points:
            print(point.time, point.value)
        ```
        """
        return await self._export(since, until, bucket_size, groups, all_variants)


class _ServerMetricsManager(_MetricsManager):
    """mdmd:namespace"""

    def __init__(self, function: _Object):
        """mdmd:hidden"""
        super().__init__(function, is_server=True)

    async def export(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        bucket_size: timedelta | None = None,
        groups: list[str] | None = None,
    ) -> ObjectMetrics:
        """Fetch metric samples for a Server.

        Defaults to all groups over the last hour. `since` defaults to one hour
        before `until`; naive datetimes use local time. The maximum range is 31
        days, with at most 500 points per series. Omit `bucket_size` for automatic
        sizing. Bounds round down to UTC bucket boundaries; the partial bucket
        at `until` is excluded. Missing observations have value None.

        Use `schema()` to discover groups and series without fetching samples.
        """
        return await self._export(since, until, bucket_size, groups)


MetricsManager = synchronize_api(_MetricsManager, target_module=__name__)
FunctionMetricsManager = synchronize_api(_FunctionMetricsManager, target_module=__name__)
ServerMetricsManager = synchronize_api(_ServerMetricsManager, target_module=__name__)
