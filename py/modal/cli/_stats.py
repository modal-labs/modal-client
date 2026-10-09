# Copyright Modal Labs 2026
"""Shared helpers for stats and metrics CLI commands."""

import csv
import io
import json
import shlex
from datetime import datetime, timedelta, timezone
from itertools import groupby
from typing import Literal

import click
from google.protobuf.timestamp_pb2 import Timestamp
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text

from modal._environments import ensure_env
from modal._functions import _Function
from modal._metrics import _FunctionMetricsManager, _metrics_request, _ServerMetricsManager
from modal._object import _get_environment_name
from modal._utils.time_utils import humanize_duration, parse_duration
from modal.client import _Client
from modal.exception import InvalidError
from modal.output import OutputManager
from modal_proto import api_pb2

from ._logs import _parse_time_arg
from .utils import HEADER_ONLY, _resolve_function_id

STATS_HEADING_STYLE = "white"
STATS_METADATA_STYLE = "bright_black"
STATS_SECTION_STYLE = "bold white"
STATS_SUCCESS_STYLE = "green"
STATS_PROGRESS_STYLE = "cyan"
STATS_WARNING_STYLE = "yellow"
STATS_PROBLEM_STYLE = "red"

_HIGH_PROBLEM_RATE = 0.05

_DEFAULT_STATS_WINDOW = timedelta(hours=1)
_DEFAULT_METRIC_PRECISION = 2
_DISPLAYED_PERCENTILES = ((5000, "p50"), (9000, "p90"), (9900, "p99"))


def stats_style(style: str, use_color: bool) -> str | Style:
    if use_color:
        return style
    if style == STATS_SECTION_STYLE:
        return "bold"
    return Style()


def success_style(count: int, total: int, use_color: bool) -> str | None:
    if use_color and total and count / total > 0.5:
        return STATS_SUCCESS_STYLE
    return None


def progress_style(count: int, use_color: bool) -> str | None:
    if use_color and count:
        return STATS_PROGRESS_STYLE
    return None


def problem_style(count: int, total: int, use_color: bool) -> str | None:
    if not use_color or count == 0:
        return None
    if not total or count / total >= _HIGH_PROBLEM_RATE:
        return STATS_PROBLEM_STYLE
    return STATS_WARNING_STYLE


def percentile_style(p50: float | None, p99: float | None, use_color: bool) -> str | Style:
    if not use_color or p50 is None or p99 is None or p99 <= 0:
        return Style()
    if p50 == 0 or p99 >= 2 * p50:
        return STATS_WARNING_STYLE
    return Style()


def _count_with_percentage(count: int, total: int, label: str) -> str:
    percentage = count / total if total else 0
    return f"{count:,} {label} ({percentage:.1%})"


def _timestamp(value: datetime) -> Timestamp:
    timestamp = Timestamp()
    timestamp.FromDatetime(value.astimezone(timezone.utc))
    return timestamp


def _percentile_table(rows: list[tuple[Text, Text, Text, Text]], use_color: bool) -> Table:
    table = Table(box=None, pad_edge=False, padding=(0, 2), header_style="")
    table.add_column("", min_width=30)
    percentile_header_style = stats_style(STATS_METADATA_STYLE, use_color)
    table.add_column(Text("p50", style=percentile_header_style), justify="right", min_width=8)
    table.add_column(Text("p90", style=percentile_header_style), justify="right", min_width=8)
    table.add_column(Text("p99", style=percentile_header_style), justify="right", min_width=8)
    for row in rows:
        table.add_row(*row)
    return table


def _percentile_name(percentile_basis_points: int) -> str:
    return f"p{percentile_basis_points / 100:g}"


def _distribution_json(distribution: api_pb2.StatsPercentileDistribution) -> dict[str, object]:
    return {
        "unit": distribution.unit,
        "percentiles": {
            _percentile_name(percentile.percentile_basis_points): percentile.value
            for percentile in distribution.percentiles
        },
    }


def _distribution_map_json(
    distributions: dict[str, api_pb2.StatsPercentileDistribution],
) -> dict[str, object]:
    return {name: _distribution_json(distribution) for name, distribution in distributions.items()}


def _metric_rows(
    distributions: dict[str, api_pb2.StatsPercentileDistribution],
    expected_order: tuple[str, ...],
    use_color: bool,
) -> list[tuple[Text, Text, Text, Text]]:
    rows: list[tuple[Text, Text, Text, Text]] = []
    ordered_names = [name for name in expected_order if name in distributions]
    ordered_names.extend(sorted(set(distributions) - set(expected_order)))
    for name in ordered_names:
        distribution = distributions[name]
        values_by_percentile = {
            percentile.percentile_basis_points: percentile.value for percentile in distribution.percentiles
        }
        if not any(basis_points in values_by_percentile for basis_points, _ in _DISPLAYED_PERCENTILES):
            continue

        values = [
            (
                f"{values_by_percentile[basis_points]:.{_DEFAULT_METRIC_PRECISION}f}"
                if basis_points in values_by_percentile
                else "—"
            )
            for basis_points, _ in _DISPLAYED_PERCENTILES
        ]
        p50 = values_by_percentile.get(5000)
        p99 = values_by_percentile.get(9900)
        rows.append(
            (
                Text(f"  {name}", style=stats_style(STATS_METADATA_STYLE, use_color)),
                Text(values[0]),
                Text(values[1]),
                Text(values[2], style=percentile_style(p50, p99, use_color)),
            )
        )
    return rows


async def _export_metrics(
    identifier: str,
    since: str | None,
    until: str | None,
    groups: tuple[str, ...],
    bucket_size: str | None,
    json_output: bool,
    csv_output: bool,
    env: str | None,
    *,
    is_server: bool,
    all_variants: bool = False,
) -> None:
    """Fetch and render metrics for the selected target."""
    object_type: Literal["Function", "Server"] = "Server" if is_server else "Function"
    if json_output and csv_output:
        raise click.UsageError("--json and --csv are mutually exclusive.")
    bucket_duration = None
    if bucket_size is not None:
        try:
            bucket_duration = parse_duration(bucket_size)
        except (ValueError, OverflowError):
            raise click.BadParameter("Use a duration such as 1m, 8h, or 1d.", param_hint="--bucket-size") from None
    now = datetime.now(timezone.utc)
    try:
        window = _metrics_request(
            _parse_time_arg(since, default=now, now=now) if since is not None else None,
            _parse_time_arg(until, default=now, now=now),
            bucket_duration,
            list(groups),
        )
    except InvalidError as exc:
        if str(exc).startswith("`bucket_size`"):
            raise click.BadParameter(str(exc), param_hint="--bucket-size") from None
        raise click.UsageError(str(exc)) from None
    since_dt = window.since.ToDatetime(tzinfo=timezone.utc)
    until_dt = window.until.ToDatetime(tzinfo=timezone.utc)
    environment_name = _get_environment_name(ensure_env(env))
    client = await _Client.from_env()
    function_id, metadata, _ = await _resolve_function_id(
        client, identifier, environment_name, object_type=object_type, command="metrics"
    )
    function: _Function = _Function._new_hydrated(function_id, client, metadata)
    manager = _ServerMetricsManager(function) if is_server else _FunctionMetricsManager(function)
    schema_groups: list[api_pb2.MetricGroupDefinition] = []
    if not json_output and not csv_output:
        try:
            schema = await manager._schema(
                since=since_dt, until=until_dt, bucket_size=bucket_duration, groups=list(groups)
            )
        except InvalidError as exc:
            raise click.UsageError(str(exc)) from None
        schema_groups = list(schema.groups)
        resolved_bucket = schema.bucket_secs
        buckets = schema.bucket_count
        start = int(since_dt.timestamp()) // resolved_bucket * resolved_bucket
        stop = int(until_dt.timestamp()) // resolved_bucket * resolved_bucket
        output = OutputManager.get()
        output.print(Text(f"Metrics schema for {function_id}{' (all variants)' if all_variants else ''}\n"))
        since_label = datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        until_label = datetime.fromtimestamp(stop, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        series_count = len({metric.name for group in schema_groups for metric in group.metrics})
        summary = Table.grid(padding=(0, 2))
        summary.add_column(no_wrap=True)
        summary.add_column()
        summary.add_row("Groups:", Text(f"{', '.join(sorted({g.name for g in schema_groups}))}"))
        summary.add_row("Range:", Text(f"[{since_label}, {until_label}) UTC"))
        summary.add_row("Bucket size:", humanize_duration(resolved_bucket))
        summary.add_row("Data size:", f"{buckets * series_count} points")
        output.print(summary)
        command = ["modal", object_type.lower(), "metrics", function_id]
        for flag, value in (("--since", since), ("--until", until), ("--bucket-size", bucket_size), ("--env", env)):
            if value is not None:
                command.extend([flag, value])
        if all_variants:
            command.append("--all-variants")
        csv_command = [*command, "--csv"]
        for group_name in groups:
            csv_command.extend(["--group", group_name])
        available = {group.name for group in schema_groups}
        example_groups = [name for name in ("requests" if is_server else "calls", "gpu") if name in available]
        if not example_groups:
            example_groups = sorted(available)[:2]
        json_command = [*command, "--json"]
        for group_name in example_groups:
            json_command.extend(["--group", group_name])
        output.print(Text("\nExport:", style="bold"))
        for example in (shlex.join(csv_command) + " > metrics.csv", shlex.join(json_command) + " > metrics.json"):
            text = Text("  " + example)
            text.highlight_regex(r"--[a-z-]+", "bold cyan")
            output.print(text)
        output.print(Text("\nNames in braces represent separate series; each listed value returns its own series."))
        for group_name, schema_rows in groupby(schema_groups, key=lambda row: row.name):
            definitions = list(schema_rows)
            group_series_count = len({metric.name for row in definitions for metric in row.metrics})
            output.print(
                Text.assemble(
                    (f"\n{group_name.upper()} ", "bold"),
                    (f"--group {group_name}", "bold cyan"),
                    f" · {group_series_count} series",
                )
            )
            rows = []
            for definition in definitions:
                if definition.display_name and definition.metrics:
                    rows.append(
                        [
                            Text(f"{definition.display_name} ({definition.metrics[0].unit})"),
                            Text(definition.description),
                        ]
                    )
                else:
                    # Schemas without display metadata render each series individually.
                    rows.extend(
                        [Text(f"{metric.name} ({metric.unit})"), Text(metric.description)]
                        for metric in definition.metrics
                    )
            table = Table(
                Column("Series (unit)", ratio=1, overflow="fold", no_wrap=False),
                Column("Description", ratio=2, overflow="fold", no_wrap=False),
                box=HEADER_ONLY,
                expand=True,
            )
            for row in rows:
                table.add_row(*row)
            output.print(table)
        return
    if isinstance(manager, _FunctionMetricsManager):
        response = await manager.export(
            since=since_dt, until=until_dt, bucket_size=bucket_duration, groups=list(groups), all_variants=all_variants
        )
    else:
        response = await manager.export(
            since=since_dt, until=until_dt, bucket_size=bucket_duration, groups=list(groups)
        )
    if csv_output:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["series_name", "unit", "time", "value"])
        for series in response.series.values():
            for point in series.points:
                writer.writerow(
                    [
                        series.name,
                        series.unit,
                        point.time.isoformat().replace("+00:00", "Z"),
                        point.value if point.value is not None else "",
                    ]
                )
        click.echo(buffer.getvalue(), nl=False)
        return
    payload = {
        "object_id": function_id,
        "since": response.since.isoformat(),
        "until": response.until.isoformat(),
        "bucket_secs": int(response.bucket_size.total_seconds()),
        "series": [
            {
                "name": series.name,
                "unit": series.unit,
                "description": series.description,
                "points": [
                    {
                        "timestamp": point.time.isoformat(),
                        "value": point.value,
                    }
                    for point in series.points
                ],
            }
            for series in response.series.values()
        ],
    }
    OutputManager.get().print_json(json.dumps(payload, allow_nan=False))
