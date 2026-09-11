# Copyright Modal Labs 2026
import json as json_lib
from datetime import datetime, timedelta, timezone
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
    STATS_SECTION_STYLE,
    _count_with_percentage,
    _distribution_map_json,
    _metric_rows,
    _percentile_table,
    _timestamp,
    problem_style,
    progress_style,
    stats_style,
    success_style,
)
from .utils import _resolve_function_id, env_option

function_cli = ModalGroup(name="function", help="Inspect Modal Functions.")

_INPUT_METRIC_ORDER = ("Execution time (s)", "End-to-end latency (s)")
_CONTAINER_METRIC_ORDER = (
    "Startup time (s)",
    "CPU Usage (cores)",
    "Memory Usage (GiB)",
    "GPU Utilization (%)",
)


def _time_range_stats_json(response: api_pb2.FunctionGetTimeRangeStatsResponse) -> dict[str, object]:
    return {
        "since": response.since.ToDatetime(tzinfo=timezone.utc).isoformat(),
        "until": response.until.ToDatetime(tzinfo=timezone.utc).isoformat(),
        "input_success_count": response.input_success_count,
        "input_failure_count": response.input_failure_count,
        "input_timeout_count": response.input_timeout_count,
        "input_running_at_end_count": response.input_running_at_end_count,
        "input_percentile_stats": _distribution_map_json(
            cast(dict[str, api_pb2.StatsPercentileDistribution], response.input_percentile_stats)
        ),
        "container_started_count": response.container_started_count,
        "container_error_count": response.container_error_count,
        "container_creating_at_end_count": response.container_creating_at_end_count,
        "container_percentile_stats": _distribution_map_json(
            cast(dict[str, api_pb2.StatsPercentileDistribution], response.container_percentile_stats)
        ),
    }


def _stats_json(
    function_id: str,
    history: api_pb2.FunctionGetTimeRangeStatsResponse,
    all_variants: bool,
) -> dict[str, object]:
    return {
        "function_id": function_id,
        "all_variants": all_variants,
        "variant_count": history.variant_count,
        **_time_range_stats_json(history),
    }


@function_cli.command("stats", no_args_is_help=True)
@click.argument("function_identifier", metavar="FUNCTION")
@click.option(
    "--since",
    default=None,
    help=(
        "Start of time range. Treated as local time "
        "if a timezone is not supplied."
        "Accepts an ISO 8601 datetime or relative time such as '2h' or '30m'."
    ),
)
@click.option(
    "--until",
    default=None,
    help=(
        "End of time range. Treated as local time if a "
        "timezone is not supplied. Accepts the same argument types as --since."
    ),
)
@click.option(
    "--all-variants",
    is_flag=True,
    default=False,
    help="Aggregate the base Function and its variants.",
)
@click.option(
    "--container",
    type=str,
    default=None,
    metavar="CONTAINER",
    help="Compute the stats only for this container. Takes precedence over --all-variants.",
)
@click.option("--no-color", is_flag=True, default=False, help="Disable colors in the output.")
@click.option("--json", "json_output", is_flag=True, default=False, help="Output stats as JSON.")
@env_option
@synchronizer.create_blocking
async def stats(
    function_identifier: str,
    since: str | None = None,
    until: str | None = None,
    all_variants: bool = False,
    container: str | None = None,
    no_color: bool = False,
    json_output: bool = False,
    *,
    env: str | None = None,
):
    """Show aggregate statistics for a modal Function.

    FUNCTION may be a Function ID or a deployed Function name in the form
    ``APP_NAME/FUNCTION_NAME``. By default, stats describe the specified Function ID.
    Pass ``--all-variants`` to aggregate across all its direct variants.

    If --since and --until are omitted, the stats are returned for the last hour.

    The available metrics and their definitions are subject to change and are provided
    on a best-effort basis. They may be delayed or incomplete. Do not rely on this command
    for autoscaler management.

    Examples:

    Show stats from the last hour using a Function ID:

    ```
    modal function stats fu-abc123
    ```

    Show stats for a deployed Function by name:

    ```
    modal function stats my-app/my-function
    ```

    Show stats across all parameterized instances of a deployed `modal.Cls`:

    ```
    modal function stats 'my-app/MyClass.*' --all-variants
    ```

    Show stats from a specific relative window:

    ```
    modal function stats my-app/my-function --since 5h --until 2h
    ```

    Show stats for an explicit time range as JSON:

    ```
    modal function stats my-app/my-function \
        --since 2026-08-28T14:00:00Z \
        --until 2026-08-28T16:00:00Z \
        --json
    ```

    Show stats for a specific container:

    ```
    modal function stats my-app/my-function --container ta-12345
    ```
    """
    now = datetime.now(timezone.utc)
    until_dt = _parse_time_arg(until, default=now)
    since_dt = _parse_time_arg(since, default=until_dt - _DEFAULT_STATS_WINDOW)
    if since_dt >= until_dt:
        raise UsageError("--since must be before --until.")

    history_duration: timedelta | None = _DEFAULT_STATS_WINDOW if since is None else None
    if since is not None and until is None:
        try:
            history_duration = parse_duration(since)
        except ValueError:
            pass

    environment_name = _get_environment_name(ensure_env(env))
    client = await _Client.from_env()
    function_id = await _resolve_function_id(
        client, function_identifier, environment_name, command="stats", include_app_id=False
    )
    req = api_pb2.FunctionGetTimeRangeStatsRequest(
        function_id=function_id,
        since=_timestamp(since_dt),
        until=_timestamp(until_dt),
        rollup=all_variants,
    )
    if container:
        req.container_id = container

    resp = await client.stub.FunctionGetTimeRangeStats(req)

    if json_output:
        OutputManager.get().print_json(json_lib.dumps(_stats_json(function_id, resp, all_variants)))
        return

    output = OutputManager.get()
    output.print("")
    heading = f"Function stats for {function_id}"
    if all_variants:
        variant_label = "variant" if resp.variant_count == 1 else "variants"
        heading += f" · all variants ({resp.variant_count:,} {variant_label})"
    use_color = not no_color
    output.print(Text(heading, style=stats_style(STATS_HEADING_STYLE, use_color)))
    output.print("")
    since_label = resp.since.ToDatetime(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    until_label = resp.until.ToDatetime(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    range_heading = ""
    if history_duration is not None:
        range_heading = f"Duration {str(history_duration)} · "
    output.print(
        Text(
            f"{range_heading}{since_label} to {until_label} UTC",
            style=stats_style(STATS_METADATA_STYLE, use_color),
        )
    )
    output.print("")

    completed_input_count = resp.input_success_count + resp.input_failure_count + resp.input_timeout_count
    input_count = completed_input_count + resp.input_running_at_end_count
    input_heading = Text("Inputs", style=stats_style(STATS_SECTION_STYLE, use_color))
    input_heading.append(" " * (46 - len("Inputs")))
    input_heading.append(f"{input_count:,} total (")
    input_heading.append(
        f"{resp.input_running_at_end_count:,} running",
        style=progress_style(resp.input_running_at_end_count, use_color),
    )
    input_heading.append(")")
    output.print(input_heading)

    input_counts = Text("  ")
    input_counts.append(
        _count_with_percentage(resp.input_success_count, completed_input_count, "succeeded"),
        style=success_style(resp.input_success_count, completed_input_count, use_color),
    )
    input_counts.append(" · ")
    input_counts.append(
        _count_with_percentage(resp.input_failure_count, completed_input_count, "failed"),
        style=problem_style(resp.input_failure_count, completed_input_count, use_color),
    )
    input_counts.append(" · ")
    input_counts.append(
        _count_with_percentage(resp.input_timeout_count, completed_input_count, "timed out"),
        style=problem_style(resp.input_timeout_count, completed_input_count, use_color),
    )
    output.print(input_counts)
    input_rows = _metric_rows(
        cast(dict[str, api_pb2.StatsPercentileDistribution], resp.input_percentile_stats),
        _INPUT_METRIC_ORDER,
        use_color,
    )
    if input_rows:
        output.print("")
        output.print(_percentile_table(input_rows, use_color))

    output.print("")
    container_count = resp.container_started_count + resp.container_error_count
    container_heading = Text("Containers", style=stats_style(STATS_SECTION_STYLE, use_color))
    container_heading.append(" " * (46 - len("Containers")))
    container_heading.append(f"{container_count:,} total (")
    container_heading.append(
        f"{resp.container_creating_at_end_count:,} creating",
        style=progress_style(resp.container_creating_at_end_count, use_color),
    )
    container_heading.append(")")
    output.print(container_heading)

    container_counts = Text("  ")
    container_counts.append(
        _count_with_percentage(resp.container_started_count, container_count, "started"),
        style=success_style(resp.container_started_count, container_count, use_color),
    )
    container_counts.append(" · ")
    container_counts.append(
        _count_with_percentage(resp.container_error_count, container_count, "errored"),
        style=problem_style(resp.container_error_count, container_count, use_color),
    )
    output.print(container_counts)

    container_rows = _metric_rows(
        cast(dict[str, api_pb2.StatsPercentileDistribution], resp.container_percentile_stats),
        _CONTAINER_METRIC_ORDER,
        use_color,
    )
    if container_rows:
        output.print("")
        output.print(_percentile_table(container_rows, use_color))


@function_cli.command("logs", no_args_is_help=True)
@click.argument("function_ref")
@click.option("-f", "--follow", is_flag=True, default=False, help="Stream log output until interrupted")
@click.option(
    "--since",
    default=None,
    help="Start of time range. Accepts ISO 8601 datetime or relative time, e.g. '1d' (1 day ago), '2h', '30m', etc.",
)
@click.option("--until", default=None, help="End of time range; accepts same argument types as --since")
@click.option("-n", "--tail", default=None, type=int, help="Show only the last N log entries")
@click.option("--search", default=None, help="Filter by search text")
@click.option("--function-call", "function_call_id", default="", help="Filter by FunctionCall ID (fc-*)")
@click.option("--container", "container_id", default="", help="Filter by Container ID (ta-*)")
@click.option("-s", "--source", default=None, help="Filter by source: 'stdout', 'stderr', or 'system'")
@click.option("--timestamps", is_flag=True, default=False, help="Prefix each line with its timestamp")
@click.option("--show-function-id", is_flag=True, default=False, help="Prefix each line with its Function ID")
@click.option("--show-function-call-id", is_flag=True, default=False, help="Prefix each line with its FunctionCall ID")
@click.option("--show-container-id", is_flag=True, default=False, help="Prefix each line with its Container ID")
@env_option
@synchronizer.create_blocking
async def logs(
    function_ref: str,
    follow: bool = False,
    since: str | None = None,
    until: str | None = None,
    tail: int | None = None,
    search: str | None = None,
    function_call_id: str = "",
    container_id: str = "",
    source: str | None = None,
    timestamps: bool = False,
    show_function_id: bool = False,
    show_function_call_id: bool = False,
    show_container_id: bool = False,
    *,
    env: str | None = None,
) -> None:
    """Fetch or stream Function logs.

    By default, this command fetches the last 100 log entries and exits. Use ``-f`` to
    live-stream logs from a running function instead. Fetch and follow are mutually exclusive.

    Examples:

    Get recent logs based on a function ID:

    ```
    modal function logs fu-12345
    ```

    Get recent logs for a currently deployed Function based on its name:

    ```
    modal function logs my-app/image-gen
    ```

    Follow (stream) logs from a running Function:

    ```
    modal function logs my-app/image-gen -f
    ```

    Fetch the last 1000 entries:

    ```
    modal function logs my-app/image-gen --tail 1000
    ```

    Fetch logs from the last 2 hours:

    ```
    modal function logs my-app/image-gen --since 2h
    ```

    Fetch logs in a specific time range:

    ```
    modal function logs my-app/image-gen --since 2026-09-01T05:00:00 --until 2026-09-01T08:00:00
    ```

    Filter the logs by source:

    ```
    modal function logs my-app/image-gen --source stderr
    ```

    Include timestamps along with function and container IDs on each line:

    ```
    modal function logs my-app/image-gen --timestamps --show-function-id --show-container-id
    ```
    """
    env = ensure_env(env)
    _validate_logs_args(follow=follow, since=since, until=until, tail=tail)

    client = await _Client.from_env()
    function_id, app_id = await _resolve_function_id(
        client, function_ref, env, object_type="Function", command="logs", include_app_id=True
    )

    prefix_fields: list[str] = []
    if show_function_id:
        prefix_fields.append("fu")
    if show_function_call_id:
        prefix_fields.append("fc")
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
        function_call_id=function_call_id,
        container_id=container_id,
        source=source,
        timestamps=timestamps,
        prefix_fields=prefix_fields,
    )
