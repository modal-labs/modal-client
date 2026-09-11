# Copyright Modal Labs 2026
from __future__ import annotations

import json as json_lib
from datetime import datetime, timezone
from typing import cast

import click
from click import UsageError
from rich.text import Text

from modal._environments import ensure_env
from modal._object import _get_environment_name
from modal._utils.async_utils import synchronizer
from modal._utils.time_utils import parse_duration
from modal.client import _Client
from modal.output import OutputManager
from modal_proto import api_pb2

from ._help import ModalGroup
from ._logs import _parse_time_arg, _run_logs_command, _validate_logs_args
from ._stats import (
    _DEFAULT_STATS_WINDOW,
    STATS_HEADING_STYLE,
    STATS_METADATA_STYLE,
    STATS_PROBLEM_STYLE,
    STATS_SECTION_STYLE,
    _count_with_percentage,
    _distribution_map_json,
    _metric_rows,
    _percentile_table,
    _timestamp,
    percentile_style,
    problem_style,
    progress_style,
    stats_style,
    success_style,
)
from .utils import _resolve_function_id, env_option

server_cli = ModalGroup(name="server", help="Manage Servers.")


@server_cli.command("logs", no_args_is_help=True)
@click.argument("server_ref")
@click.option("-f", "--follow", is_flag=True, default=False, help="Stream log output until interrupted")
@click.option(
    "--since",
    default=None,
    help="Start of time range. Accepts ISO 8601 datetime or relative time, e.g. '1d' (1 day ago), '2h', '30m', etc.",
)
@click.option("--until", default=None, help="End of time range; accepts same argument types as --since")
@click.option("-n", "--tail", default=None, type=int, help="Show only the last N log entries")
@click.option("--search", default=None, help="Filter by search text")
@click.option("--container", "container_id", default="", help="Filter by Container ID (ta-*)")
@click.option("-s", "--source", default=None, help="Filter by source: 'stdout', 'stderr', or 'system'")
@click.option("--timestamps", is_flag=True, default=False, help="Prefix each line with its timestamp")
@click.option("--show-server-id", is_flag=True, default=False, help="Prefix each line with its Server ID")
@click.option("--show-container-id", is_flag=True, default=False, help="Prefix each line with its Container ID")
@env_option
@synchronizer.create_blocking
async def logs(
    server_ref: str,
    follow: bool = False,
    since: str | None = None,
    until: str | None = None,
    tail: int | None = None,
    search: str | None = None,
    container_id: str = "",
    source: str | None = None,
    timestamps: bool = False,
    show_server_id: bool = False,
    show_container_id: bool = False,
    *,
    env: str | None = None,
) -> None:
    """Fetch or stream Server logs.

    By default, this command fetches the last 100 log entries and exits. Use ``-f`` to
    live-stream logs from a running server instead. Fetch and follow are mutually exclusive.

    Examples:

    Get recent logs based on a server ID:

    ```
    modal server logs fu-12345
    ```

    Get recent logs for a currently deployed server based on its name:

    ```
    modal server logs my-app/qwen-server
    ```

    Follow (stream) logs from a running server:

    ```
    modal server logs my-app/qwen-server -f
    ```

    Fetch the last 1000 entries:

    ```
    modal server logs my-app/qwen-server --tail 1000
    ```

    Fetch logs from the last 2 hours:

    ```
    modal server logs my-app/qwen-server --since 2h
    ```

    Fetch logs in a specific time range:

    ```
    modal server logs my-app/qwen-server --since 2026-09-01T05:00:00 --until 2026-09-01T08:00:00
    ```

    Filter the logs by source:

    ```
    modal server logs my-app/qwen-server --source stderr
    ```

    Include timestamps along with server and container IDs on each line:

    ```
    modal server logs my-app/qwen-server --timestamps --show-server-id --show-container-id
    ```
    """
    env = ensure_env(env)
    _validate_logs_args(follow=follow, since=since, until=until, tail=tail)
    client = await _Client.from_env()

    function_id, app_id = await _resolve_function_id(
        client, server_ref, env, object_type="Server", command="logs", include_app_id=True
    )

    prefix_fields: list[str] = []
    if show_server_id:
        prefix_fields.append("fu")
    if show_container_id:
        prefix_fields.append("ta")

    await _run_logs_command(
        app_id,
        follow=follow,
        since=since,
        until=until,
        tail=tail,
        search=search,
        function_id=function_id,
        container_id=container_id,
        source=source,
        timestamps=timestamps,
        prefix_fields=prefix_fields,
    )


_REQUEST_METRIC_ORDER: tuple[str, ...] = ()
_CONTAINER_METRIC_ORDER = (
    "Startup time (s)",
    "CPU usage (cores)",
    "Memory usage (GiB)",
    "GPU utilization (%)",
)
# display name, scale, precision
_INFERENCE_METRIC_FORMATS = {
    "time_to_first_token": ("Time to first token (ms)", 1000, 0),
    "inter_token_latency": ("Inter-token latency (ms)", 1000, 0),
    "end_to_end_latency": ("End-to-end latency (s)", 1, 2),
}
_INFERENCE_METRIC_ORDER = tuple(_INFERENCE_METRIC_FORMATS)
_DISPLAYED_PERCENTILES = ((5000, "p50"), (9000, "p90"), (9900, "p99"))


def _enum_value_name(value: int, enum_wrapper, prefix: str) -> str | int:
    # Preserve unknown values so older clients can serialize enums added by newer servers.
    try:
        name = enum_wrapper.Name(value)
    except ValueError:
        return value
    return name.removeprefix(prefix).lower()


def _inference_json(inference: api_pb2.ServerGetTimeRangeStatsResponse.ServerInferenceStats) -> dict[str, object]:
    return {
        "engine": _enum_value_name(inference.engine, api_pb2.LLMEngine, "LLM_ENGINE_"),
        "status": _enum_value_name(
            inference.status,
            api_pb2.ServerInferenceStatsStatus,
            "SERVER_INFERENCE_STATS_STATUS_",
        ),
        "percentile_stats": _distribution_map_json(
            cast(dict[str, api_pb2.StatsPercentileDistribution], inference.percentile_stats)
        ),
        "scalar_stats": dict(inference.scalar_stats),
    }


def _stats_json(
    function_id: str,
    history: api_pb2.ServerGetTimeRangeStatsResponse,
) -> dict[str, object]:
    return {
        "function_id": function_id,
        "since": history.since.ToDatetime(tzinfo=timezone.utc).isoformat(),
        "until": history.until.ToDatetime(tzinfo=timezone.utc).isoformat(),
        "request_count": history.request_count,
        "request_count_by_status_code": {
            f"{status_count.status_code // 100}xx": status_count.count
            for status_count in history.request_count_by_status_code
        },
        "request_rate_per_second": history.request_rate_per_second,
        "container_started_count": history.container_started_count,
        "container_error_count": history.container_error_count,
        "container_creating_at_end_count": history.container_creating_at_end_count,
        "request_percentile_stats": _distribution_map_json(
            cast(dict[str, api_pb2.StatsPercentileDistribution], history.request_percentile_stats)
        ),
        "container_percentile_stats": _distribution_map_json(
            cast(dict[str, api_pb2.StatsPercentileDistribution], history.container_percentile_stats)
        ),
        "inference": _inference_json(history.inference) if history.HasField("inference") else None,
    }


def _inference_engine_label(engine: int) -> str:
    if engine == api_pb2.LLM_ENGINE_SGLANG:
        return "SGLang"
    if engine == api_pb2.LLM_ENGINE_VLLM:
        return "vLLM"
    return "Unknown engine"


def _inference_metric_rows(
    distributions: dict[str, api_pb2.StatsPercentileDistribution],
    use_color: bool,
) -> list[tuple[Text, Text, Text, Text]]:
    rows: list[tuple[Text, Text, Text, Text]] = []
    ordered_names = [name for name in _INFERENCE_METRIC_ORDER if name in distributions]
    ordered_names.extend(sorted(set(distributions) - set(_INFERENCE_METRIC_ORDER)))
    for name in ordered_names:
        distribution = distributions[name]
        display_name, scale, precision = _INFERENCE_METRIC_FORMATS.get(name, (name, 1, 2))
        values_by_percentile = {
            percentile.percentile_basis_points: percentile.value for percentile in distribution.percentiles
        }
        if not any(basis_points in values_by_percentile for basis_points, _ in _DISPLAYED_PERCENTILES):
            continue
        values = [
            (
                f"{values_by_percentile[basis_points] * scale:.{precision}f}"
                if basis_points in values_by_percentile
                else "—"
            )
            for basis_points, _ in _DISPLAYED_PERCENTILES
        ]
        p50 = values_by_percentile.get(5000)
        p99 = values_by_percentile.get(9900)
        rows.append(
            (
                Text(f"  {display_name}", style=stats_style(STATS_METADATA_STYLE, use_color)),
                Text(values[0]),
                Text(values[1]),
                Text(values[2], style=percentile_style(p50, p99, use_color)),
            )
        )
    return rows


def _inference_throughput(inference: api_pb2.ServerGetTimeRangeStatsResponse.ServerInferenceStats) -> str:
    throughput = []
    if "input_tokens_per_second" in inference.scalar_stats:
        throughput.append(f"Input {inference.scalar_stats['input_tokens_per_second']:,.0f} tok/s")
    if "cached_input_tokens_per_second" in inference.scalar_stats:
        throughput.append(f"Cached input {inference.scalar_stats['cached_input_tokens_per_second']:,.0f} tok/s")
    if "output_tokens_per_second" in inference.scalar_stats:
        throughput.append(f"Output {inference.scalar_stats['output_tokens_per_second']:,.0f} tok/s")
    return " · ".join(throughput)


def _render_inference(history: api_pb2.ServerGetTimeRangeStatsResponse, use_color: bool) -> None:
    if not history.HasField("inference"):
        return

    output = OutputManager.get()
    inference = history.inference
    engine_label = _inference_engine_label(inference.engine)
    output.print("")
    heading = f"Inference · {engine_label}"
    output.print(Text(heading, style=stats_style(STATS_SECTION_STYLE, use_color)))
    throughput = _inference_throughput(inference)
    if throughput:
        output.print(Text(f"  {throughput}", style=stats_style(STATS_METADATA_STYLE, use_color)))

    if inference.status == api_pb2.SERVER_INFERENCE_STATS_STATUS_NO_DATA:
        output.print(
            Text(
                "  No inference metrics recorded for this range.",
                style=stats_style(STATS_METADATA_STYLE, use_color),
            )
        )
        return
    if inference.status == api_pb2.SERVER_INFERENCE_STATS_STATUS_UNAVAILABLE:
        output.print(
            Text(
                "  Inference metrics are temporarily unavailable.",
                style=stats_style(STATS_PROBLEM_STYLE, use_color),
            )
        )
        return

    percentile_rows = _inference_metric_rows(
        cast(dict[str, api_pb2.StatsPercentileDistribution], inference.percentile_stats),
        use_color,
    )
    if percentile_rows:
        output.print("")
        output.print(_percentile_table(percentile_rows, use_color))


@server_cli.command("stats", no_args_is_help=True)
@click.argument("server_identifier", metavar="SERVER")
@click.option(
    "--since",
    default=None,
    help=(
        "Start of time range. Treated as local time "
        "if a timezone is not supplied. "
        "Accepts an ISO 8601 datetime or relative time such as '2h' or '30m'."
    ),
)
@click.option(
    "--until",
    default=None,
    help=(
        "End of time range. Treated as local time if a timezone "
        "is not supplied. Accepts the same argument types as --since."
    ),
)
@click.option(
    "--container-id",
    type=str,
    default=None,
    metavar="CONTAINER_ID",
    help="Compute the stats only for this container.",
)
@click.option("--no-color", is_flag=True, default=False, help="Disable colors in the output.")
@click.option("--json", "json_output", is_flag=True, default=False, help="Output stats as JSON.")
@env_option
@synchronizer.create_blocking
async def stats(
    server_identifier: str,
    since: str | None = None,
    until: str | None = None,
    container_id: str | None = None,
    no_color: bool = False,
    json_output: bool = False,
    *,
    env: str | None = None,
):
    """Show aggregate statistics for a Modal Server.

    SERVER may be a Function ID or a deployed Server name in the form
    ``APP_NAME/SERVER_NAME``. The default time range is the most recent hour.

    The available metrics and their definitions are subject to change and are provided
    on a best-effort basis. They may be delayed or incomplete. Do not rely on this command
    for autoscaler management.

    Examples:

    Show stats for the last hour:

    ```
    modal server stats fu-abc123
    ```

    Show stats for relative window:

    ```
    modal server stats my-app/my-server --since 5h --until 2h
    ```

    Show stats for one hour window ending two hours ago:

    ```
    modal server stats my-app/my-server --until 2h
    ```

    Show stats for explicit window in json format:
    ```
    modal server stats my-app/my-server \
        --since 2026-08-28T14:00:00Z \
        --until 2026-08-28T16:00:00Z \
        --json
    ```

    Show stats for a specific container:

    ```
    modal server stats my-app/my-server --container-id ta-12345
    ```
    """
    now = datetime.now(timezone.utc)
    until_dt = _parse_time_arg(until, default=now)
    since_dt = _parse_time_arg(since, default=until_dt - _DEFAULT_STATS_WINDOW)
    if since_dt >= until_dt:
        raise UsageError("--since must be before --until.")

    history_duration = until_dt - since_dt
    if since is not None and until is None:
        try:
            history_duration = parse_duration(since)
        except ValueError:
            pass

    environment_name = _get_environment_name(ensure_env(env))
    client = await _Client.from_env()
    function_id = await _resolve_function_id(
        client,
        server_identifier,
        environment_name,
        object_type="Server",
        command="stats",
        include_app_id=False,
    )
    req = api_pb2.ServerGetTimeRangeStatsRequest(
        function_id=function_id,
        since=_timestamp(since_dt),
        until=_timestamp(until_dt),
    )
    if container_id:
        req.container_id = container_id

    history = await client.stub.ServerGetTimeRangeStats(req)

    if json_output:
        OutputManager.get().print_json(json_lib.dumps(_stats_json(function_id, history)))
        return

    output = OutputManager.get()
    output.print("")
    use_color = not no_color
    output.print(Text(f"Server stats for {function_id}", style=stats_style(STATS_HEADING_STYLE, use_color)))
    output.print("")
    since_label = history.since.ToDatetime(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    until_label = history.until.ToDatetime(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    range_heading = f"Duration {str(history_duration)} · "
    output.print(
        Text(
            f"{range_heading}{since_label} to {until_label} UTC",
            style=stats_style(STATS_METADATA_STYLE, use_color),
        )
    )
    output.print("")

    request_heading = Text("Requests", style=stats_style(STATS_SECTION_STYLE, use_color))
    request_heading.append(" " * (46 - len("Requests")))
    request_heading.append(f"{history.request_count:,} total ({history.request_rate_per_second:,.2f} req/s)")
    output.print(request_heading)
    if history.request_count_by_status_code:
        status_counts = Text("  ")
        for index, item in enumerate(sorted(history.request_count_by_status_code, key=lambda item: item.status_code)):
            if index:
                status_counts.append(" · ")
            status_family = item.status_code // 100
            style = None
            if status_family == 2:
                style = success_style(item.count, history.request_count, use_color)
            elif status_family in (4, 5):
                style = problem_style(item.count, history.request_count, use_color)
            status_counts.append(f"{status_family}xx: {item.count:,}", style=style)
        output.print(status_counts)

    request_rows = _metric_rows(
        cast(dict[str, api_pb2.StatsPercentileDistribution], history.request_percentile_stats),
        _REQUEST_METRIC_ORDER,
        use_color,
    )
    if request_rows:
        output.print("")
        output.print(_percentile_table(request_rows, use_color))

    output.print("")
    container_count = history.container_started_count + history.container_error_count
    container_heading = Text("Containers", style=stats_style(STATS_SECTION_STYLE, use_color))
    container_heading.append(" " * (46 - len("Containers")))
    container_heading.append(f"{container_count:,} total (")
    container_heading.append(
        f"{history.container_creating_at_end_count:,} creating",
        style=progress_style(history.container_creating_at_end_count, use_color),
    )
    container_heading.append(")")
    output.print(container_heading)

    container_counts = Text("  ")
    container_counts.append(
        _count_with_percentage(history.container_started_count, container_count, "started"),
        style=success_style(history.container_started_count, container_count, use_color),
    )
    container_counts.append(" · ")
    container_counts.append(
        _count_with_percentage(history.container_error_count, container_count, "errored"),
        style=problem_style(history.container_error_count, container_count, use_color),
    )
    output.print(container_counts)
    container_rows = _metric_rows(
        cast(dict[str, api_pb2.StatsPercentileDistribution], history.container_percentile_stats),
        _CONTAINER_METRIC_ORDER,
        use_color,
    )
    if container_rows:
        output.print("")
        output.print(_percentile_table(container_rows, use_color))

    _render_inference(history, use_color)
