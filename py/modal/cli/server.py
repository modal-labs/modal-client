# Copyright Modal Labs 2026
from __future__ import annotations

import dataclasses
import json as json_lib
import sys
from datetime import datetime, timezone
from typing import AsyncGenerator, cast

import click
from click import UsageError
from rich.table import Column, Table
from rich.text import Text

from modal._environments import ensure_env
from modal._object import _get_environment_name
from modal._server import _Server
from modal._utils.async_utils import async_map_ordered, synchronizer
from modal._utils.time_utils import parse_duration
from modal.cli.utils import humanize_filesize
from modal.client import _Client
from modal.exception import NotFoundError
from modal.output import OutputManager
from modal.secret import _Secret
from modal.types import ServerAutoscalerSettings
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
from .utils import (
    HEADER_ONLY,
    _resolve_function_id,
    display_table,
    env_option,
    grouped_utc_timestamp,
)

server_cli = ModalGroup(name="server", help="Manage Servers.")

_DEFAULT_REQUEST_TAIL = 10
_MAX_REQUEST_TAIL = 1000


def _server_request_status_cell(status: int, no_color: bool = False) -> Text:
    return Text(str(status), style="red") if status >= 400 and not no_color else Text(str(status))


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

    function_id, metadata, _ = await _resolve_function_id(client, server_ref, env, object_type="Server", command="logs")

    prefix_fields: list[str] = []
    if show_server_id:
        prefix_fields.append("fu")
    if show_container_id:
        prefix_fields.append("ta")

    await _run_logs_command(
        metadata.app_id,
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


@server_cli.command("requests", no_args_is_help=True)
@click.argument("server_identifier", metavar="SERVER")
@click.option(
    "-n",
    "--tail",
    type=click.IntRange(min=1, max=_MAX_REQUEST_TAIL),
    default=_DEFAULT_REQUEST_TAIL,
    show_default=True,
    help="Show up to the last N Server requests.",
)
@click.option("--json", "json_output", is_flag=True, default=False, help="Output requests as JSON.")
@click.option("--no-color", "no_color", is_flag=True, default=False, help="Disable colors in the output.")
@env_option
@synchronizer.create_blocking
async def requests(
    server_identifier: str,
    tail: int = _DEFAULT_REQUEST_TAIL,
    json_output: bool = False,
    no_color: bool = False,
    *,
    env: str | None = None,
) -> None:
    """Show recent requests handled by a Modal Server.

    SERVER may be a Function ID or a deployed Server name in the form
    ``APP_NAME/SERVER_NAME``.

    Examples:

    ```
    modal server requests my-app/my-server
    ```

    ```
    modal server requests my-app/my-server --tail 500
    ```

    Disable color in the output:

    ```
    modal server requests my-app/my-server --no-color
    ```
    """
    environment_name = _get_environment_name(ensure_env(env))
    client = await _Client.from_env()
    function_id, _, _ = await _resolve_function_id(
        client,
        server_identifier,
        environment_name,
        object_type="Server",
        command="requests",
    )
    response = await client._stub.ServerRequestFetch(
        api_pb2.ServerRequestFetchRequest(
            function_id=function_id, tail=api_pb2.ServerRequestFetchRequest.Tail(count=tail)
        )
    )

    if json_output:
        OutputManager.get().print_json(
            json_lib.dumps(
                [
                    {
                        "timestamp": request.timestamp.ToJsonString(),
                        "route": request.route,
                        "container_id": request.container_id,
                        "duration_seconds": request.duration_seconds,
                        "status": request.status,
                    }
                    for request in response.requests
                ]
            )
        )
        return

    rows: list[list[Text | str]] = []
    previous_request_date = None
    for request in response.requests:
        timestamp, previous_request_date = grouped_utc_timestamp(request.timestamp, previous_request_date)
        rows.append(
            [
                Text(timestamp),
                Text(request.route),
                Text(request.container_id),
                Text(f"{request.duration_seconds:.2f}"),
                _server_request_status_cell(request.status, no_color=no_color),
            ]
        )
    output = OutputManager.get()
    output.print("")
    output.print(Text(f"Server requests for {function_id}"))
    output.print(
        Text(
            f"Displaying {len(rows):,} {'row' if len(rows) == 1 else 'rows'}",
            style=STATS_METADATA_STYLE if not no_color else "",
        )
    )
    output.print("")
    display_table(
        [
            Column("Timestamp (UTC)", justify="right"),
            "Route",
            Column("Container", width=29),
            Column("Duration (s)", justify="right"),
            "Status",
        ],
        rows,
        table_box=HEADER_ONLY,
        border_style=STATS_METADATA_STYLE if not no_color else None,
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


def _enum_value_name(value: int, enum_wrapper, prefix: str) -> str:
    # Preserve unknown values so older clients can serialize enums added by newer servers.
    try:
        name = enum_wrapper.Name(value)
    except ValueError:
        return "unrecognized"
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
    function_id, _, _ = await _resolve_function_id(
        client,
        server_identifier,
        environment_name,
        object_type="Server",
        command="stats",
    )
    req = api_pb2.ServerGetTimeRangeStatsRequest(
        function_id=function_id,
        since=_timestamp(since_dt),
        until=_timestamp(until_dt),
    )
    if container_id:
        req.container_id = container_id

    history = await client._stub.ServerGetTimeRangeStats(req)

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


@server_cli.command("info", no_args_is_help=True)
@click.argument("server_identifier", metavar="SERVER")
@click.option("--json", "json", is_flag=True, default=False, help="Output as JSON.")
@env_option
@synchronizer.create_blocking
async def info(
    server_identifier: str,
    json: bool = False,
    *,
    env: str | None = None,
):
    """Show information about a given Modal Server.

    SERVER can either be a Function ID (`fu-...`) or a deployed Server name in the format
    `APP_NAME/SERVER_NAME`. The output of this command includes information about any resources
    requested by this Server, any scheduling/autoscaling settings, any mounted Volumes or Buckets,
    and any HTTP settings.

    Examples:

    Providing a Server ID directly:

    ```
    modal server info fu-0123456789abcdefghijkl
    ```

    Referring to a Server within a deployed App:

    ```
    modal server info hello-world-app/test_server
    ```
    """
    client = await _Client.from_env()
    environment_name = _get_environment_name(ensure_env(env))
    tty = sys.stdout.isatty()

    server_id, handle_metadata, _ = await _resolve_function_id(
        client,
        server_identifier,
        environment_name,
        object_type="Server",
        command="info",
    )

    server = _Server._new_from_function(server_id, client, handle_metadata)
    info = await server.info()

    autoscaler_response = await client._stub.FunctionGetSchedulingParams(
        api_pb2.FunctionGetSchedulingParamsRequest(function_id=server_id)
    )
    autoscaler_settings = ServerAutoscalerSettings._from_proto(autoscaler_response.autoscaler_configuration.settings)

    web_url = await server.get_url()
    output_manager = OutputManager.get()

    if json:
        info_dict = dataclasses.asdict(info)
        info_dict = info_dict | dataclasses.asdict(autoscaler_settings)
        info_dict["web_url"] = web_url

        output_manager.print_json(json_lib.dumps(info_dict))
        return

    not_configured = "-"
    enabled = "Enabled"
    disabled = "Disabled"

    rows: list[str | Text | tuple[str | Text, str | Text]] = []

    rows.append(("Server Name:", handle_metadata.function_name))
    rows.append(("Function ID:", server_id))
    rows.append(("App ID:", handle_metadata.app_id))
    rows.append(("Image ID:", str(info.image_info.image_id)))

    # --- HTTP ---
    rows.append("HTTP Info:")
    if web_url:
        rows.append(("  URL:", web_url))
    rows.append(("  Port:", not_configured if not info.http_info.port else str(info.http_info.port)))
    rows.append(("  Authentication:", disabled if info.http_info.unauthenticated else enabled))
    rows.append(("  HTTP/2:", disabled if not info.http_info.h2_enabled else enabled))
    rows.append(
        (
            "  Proxy Region(s):",
            not_configured if len(info.http_info.proxy_regions) == 0 else " | ".join(info.http_info.proxy_regions),
        )
    )
    rows.append(("  Sessioned:", not_configured if not info.sessioned else enabled))

    # --- Resources ---
    rows.append("Resources:")
    rows.append(
        (
            "  CPU:",
            not_configured
            if info.cpu is None
            else f"{info.cpu} core(s)"
            if isinstance(info.cpu, (int, float))
            else f"{info.cpu[0]} - {info.cpu[1]} core(s)",
        )
    )
    rows.append(
        (
            "  Memory:",
            not_configured
            if info.memory_mib is None
            else humanize_filesize(info.memory_mib << 20)
            if isinstance(info.memory_mib, int)
            else f"{humanize_filesize(info.memory_mib[0] << 20)} - {humanize_filesize(info.memory_mib[1] << 20)}",
        )
    )
    rows.append(
        (
            "  Ephemeral Disk:",
            not_configured if info.ephemeral_disk_mib is None else (humanize_filesize(info.ephemeral_disk_mib << 20)),
        )
    )
    rows.append(
        (
            "  GPU(s):",
            not_configured
            if len(info.gpus) == 0
            else " | ".join([f"{gpu_type} x {count}" for gpu_type, count in info.gpus]),
        )
    )

    # --- Autoscaling ---
    rows.append("Autoscaling:")
    rows.append(
        (
            "  Min/Max/Buffer Containers:",
            " / ".join(
                [
                    not_configured
                    if autoscaler_settings.min_containers is None
                    else str(autoscaler_settings.min_containers),
                    not_configured
                    if autoscaler_settings.max_containers is None
                    else str(autoscaler_settings.max_containers),
                    not_configured
                    if autoscaler_settings.buffer_containers is None
                    else str(autoscaler_settings.buffer_containers),
                ]
            ),
        )
    )

    rows.append(
        (
            "  Scaleup Window:",
            not_configured
            if autoscaler_settings.scaleup_window is None
            else f"{autoscaler_settings.scaleup_window} seconds",
        )
    )
    rows.append(
        (
            "  Scaledown Window:",
            not_configured
            if autoscaler_settings.scaledown_window is None
            else f"{autoscaler_settings.scaledown_window} seconds",
        )
    )
    rows.append(
        (
            "  Target Concurrency:",
            not_configured
            if not autoscaler_settings.target_concurrency
            else f"{autoscaler_settings.target_concurrency:.1f} requests / container",
        )
    )

    rows.append("Execution:")
    rows.append(("  Timeout:", f"{info.timeout} seconds"))
    rows.append(("  Max Retries:", not_configured if info.max_retries is None else str(info.max_retries)))

    if info.batching_info:
        rows.append("Batching:")
        rows.append(("  Max Batch Size:", str(info.batching_info.max_batch_size)))
        rows.append(("  Wait Time:", f"{info.batching_info.wait_ms} milliseconds"))

    # --- Scheduling ---
    rows.append("Scheduling:")
    rows.append(
        ("  Compute Region(s):", not_configured if not info.compute_regions else " | ".join(info.compute_regions))
    )
    rows.append(("  Nonpreemptible Capacity:", not_configured if not info.nonpreemptible else enabled))
    rows.append(("  Cloud Provider:", not_configured if info.cloud is None else (info.cloud)))

    if info.cluster_info:
        rows.append("Clustering:")
        rows.append(("  Cluster Size:", str(info.cluster_info.size)))
        rows.append(("  RDMA:", enabled if info.cluster_info.rdma else disabled))
        if info.cluster_info.fabric_size is not None:
            rows.append(("  Fabric Size:", str(info.cluster_info.fabric_size)))

    # --- Mounts ---
    if info.volumes:
        rows.append("Volume Mounts:")
        for mount_path, volume in info.volumes.items():
            assert volume.volume_id is not None

            extra = []
            if volume.read_only:
                extra.append("Read-Only")
            if volume.sub_path:
                extra.append(f"Sub-Path: {volume.sub_path}")

            if extra:
                volume_text = f"{volume.volume_id} ({', '.join(extra)})"
            else:
                volume_text = volume.volume_id

            rows.append((f"  {mount_path}", volume_text))

    if info.cloud_bucket_mounts:
        rows.append("Cloud Bucket Mounts:")
        for mount_path, cbm in info.cloud_bucket_mounts.items():
            extra = []
            if cbm.read_only:
                extra.append("Read-Only")
            if cbm.key_prefix:
                extra.append(f"Prefix: {cbm.key_prefix}")

            if extra:
                cbm_text = f"{cbm.bucket_name} ({', '.join(extra)})"
            else:
                cbm_text = cbm.bucket_name

            rows.append((f"  {mount_path}", cbm_text))

    # --- Secrets ---
    if info.secrets:
        rows.append("Secrets:")

        async def _hydrate(secret_id: str) -> tuple[str, _Secret | None]:
            try:
                s = _Secret._from_id(secret_id)
                await s.hydrate()
            except NotFoundError:
                return secret_id, None

            return secret_id, s

        async def _secret_iter() -> AsyncGenerator[str, None]:
            for secret_id in info.secrets:
                yield secret_id

        async for secret_id, secret in async_map_ordered(
            _secret_iter(),
            _hydrate,
            10,
        ):
            if secret is None:
                rows.append((f"  {secret_id}", "[DELETED]"))
                continue

            secret_identifier = secret.object_id
            if secret._name:
                secret_identifier = secret._name
                secret_env = secret._get_metadata().environment_name

                if secret_env != environment_name:
                    secret_identifier = f"{secret._name} ({secret_env})"

            keys = await secret._get_keys()
            display_keys = sorted(keys)[:5]
            if len(display_keys) < len(keys):
                display_keys.append(f"({len(keys) - len(display_keys)} keys omitted)")

            rows.append((f"  {secret_identifier}", ", ".join(display_keys)))

    for row in rows:
        t = Table().grid(padding=(0, 0, 3, 3), expand=True)

        if not isinstance(row, tuple):
            if isinstance(row, str):
                row = Text(row)

            if not tty:
                output_manager.print(row)
            else:
                t.add_row(row)
                output_manager.print(t)

            continue

        left, right = row

        if not tty:
            output_manager.print(Text(f"{left} {right}"))
            continue

        if isinstance(left, str):
            left = Text(left, overflow="fold")
        if isinstance(right, str):
            right = Text(right, overflow="fold", justify="right")

        t.add_row(left, right)
        output_manager.print(t)
