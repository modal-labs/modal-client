# Copyright Modal Labs 2026
"""Shared helpers for stats CLI commands."""

from datetime import datetime, timedelta, timezone

from google.protobuf.timestamp_pb2 import Timestamp
from rich.style import Style
from rich.table import Table
from rich.text import Text

from modal_proto import api_pb2

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
