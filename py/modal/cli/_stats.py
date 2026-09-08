# Copyright Modal Labs 2026
"""Shared color styling for stats CLI commands."""

from rich.style import Style

STATS_HEADING_STYLE = "white"
STATS_METADATA_STYLE = "bright_black"
STATS_SECTION_STYLE = "bold white"
STATS_SUCCESS_STYLE = "green"
STATS_PROGRESS_STYLE = "cyan"
STATS_WARNING_STYLE = "yellow"
STATS_PROBLEM_STYLE = "red"

_HIGH_PROBLEM_RATE = 0.05


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
