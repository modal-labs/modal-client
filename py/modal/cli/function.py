# Copyright Modal Labs 2026
import dataclasses
import json as json_lib
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, cast

import click
from click import UsageError
from rich.table import Column, Table
from rich.text import Text

from modal._environments import ensure_env
from modal._function_variants import _FunctionOptionsInfo, _list_function_variants
from modal._functions import _Function
from modal._object import _get_environment_name
from modal._utils.async_utils import async_map_ordered, synchronizer
from modal._utils.time_utils import parse_duration
from modal.client import _Client
from modal.exception import NotFoundError
from modal.output import OutputManager
from modal.retries import Retries
from modal.secret import _Secret
from modal.types import CloudBucketMountInfo, FunctionAutoscalerSettings, FunctionInfo, VolumeMountInfo
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
from .utils import (
    HEADER_ONLY,
    _resolve_function_id,
    display_table,
    env_option,
    grouped_utc_timestamp,
    humanize_filesize,
)

function_cli = ModalGroup(name="function", help="Inspect Modal Functions.")

_INPUT_METRIC_ORDER = ("Execution time (s)", "End-to-end latency (s)")
_CONTAINER_METRIC_ORDER = (
    "Startup time (s)",
    "CPU Usage (cores)",
    "Memory Usage (GiB)",
    "GPU Utilization (%)",
)
_DEFAULT_CALL_TAIL = 10
_MAX_CALL_TAIL = 1000
_FAILURE_STATUSES = {
    api_pb2.FUNCTION_CALL_INPUT_STATUS_FAILURE: "Failure",
    api_pb2.FUNCTION_CALL_INPUT_STATUS_TIMEOUT: "Timeout",
    api_pb2.FUNCTION_CALL_INPUT_STATUS_TERMINATED: "Terminated",
    api_pb2.FUNCTION_CALL_INPUT_STATUS_INIT_FAILURE: "Failure",
    api_pb2.FUNCTION_CALL_INPUT_STATUS_INTERNAL_FAILURE: "Failure",
    api_pb2.FUNCTION_CALL_INPUT_STATUS_IDLE_TIMEOUT: "Timeout (idle)",
    api_pb2.FUNCTION_CALL_INPUT_STATUS_MEMORY_MANAGER_EVICTION: "Evicted (over memory request)",
}


def _optional_duration(value: float, present: bool) -> str:
    if not present:
        return "—"
    return f"{value:.2f}"


def _function_call_status(status: api_pb2.FunctionCallInputStatus.ValueType, json_output: bool = False) -> str | int:
    try:
        name = api_pb2.FunctionCallInputStatus.Name(status)
    except ValueError:
        return status
    label = name.removeprefix("FUNCTION_CALL_INPUT_STATUS_").replace("_", " ")
    if json_output:
        return label.lower()
    return _FAILURE_STATUSES.get(status, label.title())


def _function_call_status_cell(status: api_pb2.FunctionCallInputStatus.ValueType, no_color: bool = False) -> Text:
    style = None
    if status == api_pb2.FUNCTION_CALL_INPUT_STATUS_PENDING:
        style = "yellow"
    elif status == api_pb2.FUNCTION_CALL_INPUT_STATUS_RUNNING:
        style = "green"
    elif status in _FAILURE_STATUSES:
        style = "red"
    label = str(_function_call_status(status))
    return Text(label, style=style) if style and not no_color else Text(label)


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
    function_id, _, _ = await _resolve_function_id(client, function_identifier, environment_name, command="stats")
    req = api_pb2.FunctionGetTimeRangeStatsRequest(
        function_id=function_id,
        since=_timestamp(since_dt),
        until=_timestamp(until_dt),
        rollup=all_variants,
    )
    if container:
        req.container_id = container

    resp = await client._stub.FunctionGetTimeRangeStats(req)

    if json_output:
        OutputManager.get().print_json(json_lib.dumps(_stats_json(function_id, resp, all_variants)))
        return

    output = OutputManager.get()
    output.print("")
    heading = f"Function stats for {function_id}"
    if resp.variant_count == 0 and not all_variants:
        variant_label = ""
    else:
        variant_label = f" ({resp.variant_count:,} {'variant' + ('s' if resp.variant_count != 1 else '')})"
    heading += f"{' · all variants' if all_variants else ''}{variant_label}"
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


@function_cli.command("calls", no_args_is_help=True)
@click.argument("function_identifier", metavar="FUNCTION")
@click.option(
    "-n",
    "--tail",
    type=click.IntRange(min=1, max=_MAX_CALL_TAIL),
    default=_DEFAULT_CALL_TAIL,
    show_default=True,
    help="Show up to the last N Function inputs.",
)
@click.option(
    "--all-variants",
    is_flag=True,
    default=False,
    help="Include inputs from the base Function and all its variants.",
)
@click.option(
    "--show-function-call-id",
    is_flag=True,
    default=False,
    help="Include the Function Call ID in table output.",
)
@click.option("--json", "json_output", is_flag=True, default=False, help="Output calls as JSON.")
@click.option("--no-color", "no_color", is_flag=True, default=False, help="Disable colors in the output.")
@env_option
@synchronizer.create_blocking
async def calls(
    function_identifier: str,
    tail: int = _DEFAULT_CALL_TAIL,
    all_variants: bool = False,
    show_function_call_id: bool = False,
    json_output: bool = False,
    no_color: bool = False,
    *,
    env: str | None = None,
) -> None:
    """Show recent inputs for a Modal Function.

    FUNCTION may be a Function ID or a deployed Function name in the form
    ``APP_NAME/FUNCTION_NAME``. Each unique input corresponds to one entry
    in the output.

    Examples:

    ```
    modal function calls my-app/my-function
    ```

    Show recent calls across all variants of a Cls:

    ```
    modal function calls 'my-app/MyClass.*' --all-variants --tail 500
    ```

    Disable color in the output:

    ```
    modal function calls my-app/my-function --no-color
    ```
    """
    environment_name = _get_environment_name(ensure_env(env))
    client = await _Client.from_env()
    function_id, _, _ = await _resolve_function_id(
        client,
        function_identifier,
        environment_name,
        command="calls",
    )
    response = await client._stub.FunctionCallFetch(
        api_pb2.FunctionCallFetchRequest(
            function_id=function_id,
            tail=api_pb2.FunctionCallFetchRequest.Tail(count=tail),
            all_variants=all_variants,
        )
    )

    if json_output:
        OutputManager.get().print_json(
            json_lib.dumps(
                [
                    {
                        "function_call_id": call.function_call_id,
                        "service_method_name": (
                            call.service_method_name if call.HasField("service_method_name") else None
                        ),
                        "enqueued_at": (call.enqueued_at.ToJsonString() if call.HasField("enqueued_at") else None),
                        "started_at": (call.started_at.ToJsonString() if call.HasField("started_at") else None),
                        "container_id": call.container_id if call.HasField("container_id") else None,
                        "startup_time_seconds": (
                            call.startup_time_seconds if call.HasField("startup_time_seconds") else None
                        ),
                        "execution_time_seconds": (
                            call.execution_time_seconds if call.HasField("execution_time_seconds") else None
                        ),
                        "status": _function_call_status(call.status, True),
                    }
                    for call in response.function_call_inputs
                ]
            )
        )
        return

    show_service_method_name = bool(response.function_call_inputs) and all(
        call.HasField("service_method_name") for call in response.function_call_inputs
    )
    rows: list[list[Text | str]] = []
    previous_enqueued_date = None
    for call in response.function_call_inputs:
        if call.HasField("enqueued_at"):
            enqueued_at, enqueued_date = grouped_utc_timestamp(call.enqueued_at, previous_enqueued_date)
            previous_enqueued_date = enqueued_date
        else:
            enqueued_at = "—"
            enqueued_date = None

        if call.HasField("started_at") and enqueued_date:
            started_secs = (
                call.started_at.ToDatetime(tzinfo=timezone.utc) - call.enqueued_at.ToDatetime(tzinfo=timezone.utc)
            ).total_seconds()
            started_at = f"{started_secs:.2f}"
        else:
            started_at = "—"

        row: list[Text | str] = [Text(enqueued_at), Text(started_at)]
        if show_function_call_id:
            row.append(Text(call.function_call_id or "—"))
        if show_service_method_name:
            row.append(Text(call.service_method_name))
        row.extend(
            [
                Text(
                    call.container_id if call.HasField("container_id") else "—",
                ),
                Text(
                    _optional_duration(call.execution_time_seconds, call.HasField("execution_time_seconds")),
                ),
                _function_call_status_cell(call.status, no_color=no_color),
            ]
        )
        rows.append(row)

    title = f"Function calls for {function_id}"
    if all_variants:
        title += " · all variants"
    output = OutputManager.get()
    output.print("")
    output.print(Text(title))
    output.print(
        Text(
            f"Displaying {len(rows):,} {'row' if len(rows) == 1 else 'rows'}",
            style=STATS_METADATA_STYLE if not no_color else "",
        )
    )
    output.print("")
    columns: list[str | Column] = [
        Column("Enqueued (UTC)", no_wrap=True, justify="right"),
        Column("Queuing (s)", justify="right"),
    ]
    if show_function_call_id:
        columns.append("Function Call ID")
    if show_service_method_name:
        columns.append("Method")
    columns.extend(
        [
            Column("Container", width=29),
            Column("Execution (s)", justify="right"),
            "Status",
        ]
    )
    display_table(
        columns,
        rows,
        table_box=HEADER_ONLY,
        border_style=STATS_METADATA_STYLE if not no_color else "",
    )


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
@click.option("--all-variants", is_flag=True, default=False, help="Include logs from the base and all its variants.")
@env_option
@synchronizer.create_blocking
async def logs(
    function_ref: str,
    follow: bool = False,
    all_variants: bool = False,
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

    By default, logs are limited to the specified ID. Pass ``--all-variants`` to
    include the base and all its variants, even when specifying a variant ID.

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
    function_id, metadata, _ = await _resolve_function_id(
        client, function_ref, env, object_type="Function", command="logs"
    )
    app_id = metadata.app_id

    prefix_fields: list[str] = []
    if show_function_id:
        prefix_fields.append("fu")
    if show_function_call_id:
        prefix_fields.append("fc")
    if show_container_id:
        prefix_fields.append("ta")

    await _run_logs_command(
        metadata.app_id,
        follow=follow,
        since=since,
        until=until,
        tail=tail,
        search=search,
        function_id=metadata.base_function_id or function_id,
        parametrized_function_id="" if all_variants else function_id,
        function_call_id=function_call_id,
        container_id=container_id,
        source=source,
        timestamps=timestamps,
        prefix_fields=prefix_fields,
    )


def _overridden_options(options: _FunctionOptionsInfo | None) -> list[tuple[str, Any]]:
    """Return the settings a variant overrides, keyed by the builder-method parameter name."""
    if options is None:
        return []
    fields = ((field.name, getattr(options, field.name)) for field in dataclasses.fields(options))
    return [(name, value) for name, value in fields if value is not None]


def _variant_overrides_json(options: _FunctionOptionsInfo | None) -> dict[str, Any]:
    """Return the overridden settings in a form that can be serialized as JSON."""
    overrides: dict[str, Any] = {}
    for name, value in _overridden_options(options):
        if isinstance(value, Retries):
            value = {
                "max_retries": value.max_retries,
                "backoff_coefficient": value.backoff_coefficient,
                "initial_delay": value.initial_delay.total_seconds(),
                "max_delay": value.max_delay.total_seconds(),
            }
        elif isinstance(value, tuple):
            value = list(value)
        elif name in ("cloud_bucket_mounts", "volumes"):
            value = {path: dataclasses.asdict(mount) for path, mount in value.items()}
        overrides[name] = value
    return overrides


def _format_bucket_mount(mount: CloudBucketMountInfo) -> str:
    """Render a cloud bucket mount as a URI, which is the part that identifies it at a glance."""
    uri = f"{mount.bucket_type}://{mount.bucket_name}"
    if mount.key_prefix:
        uri = f"{uri}/{mount.key_prefix}"
    return f"{uri} (ro)" if mount.read_only else uri


def _format_volume_mount(mount: VolumeMountInfo) -> str:
    """Render a volume mount as its ID plus whatever `.with_mount_options()` changed about it."""
    volume = mount.volume_id or mount.name or ""
    if mount.sub_path:
        volume = f"{volume}:{mount.sub_path}"
    return f"{volume} (ro)" if mount.read_only else volume


def _format_override(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_format_override(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_override(v) for v in value) + "]"
    return str(value)


def _variant_cell(text: str, style: str = "") -> Text:
    """Render one table cell that is truncated rather than wrapped, so one variant stays in one row."""
    return Text(text, style=style, no_wrap=True, overflow="ellipsis")


def _format_parameter(value: Any) -> str:
    if isinstance(value, bytes):
        # A bytes parameter is an opaque payload; its size is the only part that reads well in a table.
        return f"<bytes: {humanize_filesize(len(value))}>"
    return repr(value)


def _parameters_cell(parameters: dict[str, Any] | None) -> Text:
    if parameters is None:
        return _variant_cell("unavailable", style="dim")
    return _variant_cell(", ".join(f"{name}={_format_parameter(value)}" for name, value in parameters.items()))


def _options_cell(options: _FunctionOptionsInfo | None) -> Text:
    parts = []
    for name, value in _overridden_options(options):
        if isinstance(value, Retries):
            # The retry count is the useful part of a retry policy at a glance.
            value = value.max_retries
        elif name == "cloud_bucket_mounts":
            value = {path: _format_bucket_mount(mount) for path, mount in value.items()}
        elif name == "volumes":
            value = {path: _format_volume_mount(mount) for path, mount in value.items()}
        parts.append(f"{name}={_format_override(value)}")
    return _variant_cell(", ".join(parts))


@function_cli.command("variants", no_args_is_help=True)
@click.argument("function_ref", metavar="FUNCTION")
@click.option(
    "-n",
    "--limit",
    "limit",
    default=200,
    show_default=True,
    type=int,
    help="Show at most N variants, those running the most containers first. Use 0 to list every variant, newest first.",
)
@click.option("--json", "json", is_flag=True, default=False, help="Output as JSON.")
@env_option
@synchronizer.create_blocking
async def variants(
    function_ref: str,
    limit: int = 200,
    json: bool = False,
    env: str | None = None,
):
    """List the variants of a modal Function.

    Variants are the parameterized instances of a `modal.Cls` and the Functions created with
    `.with_options()`. FUNCTION may be a Function ID or a deployed Function name in the form
    ``APP_NAME/FUNCTION_NAME``.

    By default, the busiest variants are listed first. If you ask for more variants than can be
    ranked, or for all of them, they are listed newest first instead.

    Examples:

    List the busiest variants of a deployed Function:

    ```
    modal function variants my-app/my-function
    ```

    List the parameterized instances of a deployed `modal.Cls`:

    ```
    modal function variants 'my-app/MyClass.*'
    ```

    Show only the ten busiest variants:

    ```
    modal function variants my-app/my-function --limit 10
    ```

    List every variant of a Function ID as JSON:

    ```
    modal function variants fu-abc123 --limit 0 --json
    ```
    """
    if limit < 0:
        raise UsageError("--limit cannot be negative.")

    environment_name = _get_environment_name(ensure_env(env))
    client = await _Client.from_env()
    function_id, handle_metadata, _ = await _resolve_function_id(
        client,
        function_ref,
        environment_name,
        command="variants",
    )
    # A variant has no variants of its own, so resolving one lists the family it belongs to.
    base_function_id = handle_metadata.base_function_id or function_id
    effective_limit = limit or None
    listing = await _list_function_variants(client, base_function_id, limit=effective_limit)
    function_variants = listing.variants
    show_parameters = bool(handle_metadata.method_handle_metadata)

    output = OutputManager.get()
    if json:
        payload: list[dict[str, Any]] = []
        for variant in function_variants:
            entry: dict[str, Any] = {"function_id": variant.function_id}
            if show_parameters:
                entry["parameters"] = variant.parameters
            entry["options"] = _variant_overrides_json(variant.options)
            payload.append(entry)
        # Parameter values may include bytes, which have no JSON representation of their own.
        output.print_json(json_lib.dumps(payload, default=repr))
        return

    if not function_variants:
        output.print(f"No variants found for {base_function_id}.")
        return

    rows: list[list[Text | str | None]] = []
    for variant in function_variants:
        options_cell = _options_cell(variant.options)
        if show_parameters:
            rows.append(
                [
                    variant.function_id,
                    _parameters_cell(variant.parameters),
                    options_cell,
                ]
            )
        else:
            rows.append([variant.function_id, options_cell])

    id_column = Column("Function ID", no_wrap=True)
    columns = (
        [id_column, Column("Parameters"), Column("Options")] if show_parameters else [id_column, Column("Options")]
    )
    ordering = f"{len(rows)} busiest" if listing.ordered_by_task_count else f"{len(rows)} newest"
    display_table(columns, rows, title=f"Variants of {base_function_id} · {ordering}")

    if effective_limit is not None and len(function_variants) == effective_limit:
        output.print(
            Text(
                f"Showing {effective_limit} variants; use --limit 0 to list every variant.",
                style="dim",
            )
        )


@function_cli.command("info", no_args_is_help=True)
@click.argument("function_identifier", metavar="FUNCTION")
@click.option("--json", "json", is_flag=True, default=False, help="Output as JSON.")
@env_option
@synchronizer.create_blocking
async def info(
    function_identifier: str,
    json: bool = False,
    *,
    env: str | None = None,
):
    """Show information about a given Modal Function.

    FUNCTION can either be a Function ID (`fu-...`) or a deployed Function name in the format
    `APP_NAME/FUNCTION_NAME`. The output of this command includes information about any resources
    requested by this Function, any scheduling/autoscaling settings, any mounted Volumes or Buckets,
    and any HTTP settings.

    Examples:

    Providing a Function ID directly:

    ```
    modal function info fu-0123456789abcdefghijkl
    ```

    Referring to a Function within a deployed App:

    ```
    modal function info hello-world-app/test_web_function
    ```
    """
    tty = sys.stdout.isatty()

    client = await _Client.from_env()
    environment_name = _get_environment_name(ensure_env(env))

    function_id, handle_metadata, function_proto = await _resolve_function_id(
        client,
        function_identifier,
        environment_name,
        object_type="Function",
        command="info",
    )

    if function_proto.is_server:
        raise UsageError(f"'{function_identifier}' is a Server.")

    f: _Function = _Function._new_hydrated(function_id, client, handle_metadata=handle_metadata)
    info = FunctionInfo._from_function_proto(function_proto)

    autoscaler_response = await client._stub.FunctionGetSchedulingParams(
        api_pb2.FunctionGetSchedulingParamsRequest(function_id=f.object_id)
    )
    autoscaler_settings = FunctionAutoscalerSettings._from_proto(autoscaler_response.autoscaler_configuration.settings)

    output_manager = OutputManager.get()

    if json:
        info_dict = dataclasses.asdict(info)
        info_dict = info_dict | dataclasses.asdict(autoscaler_settings)
        info_dict.pop("_http_info")
        info_dict.pop("_sessioned")

        output_manager.print_json(json_lib.dumps(info_dict))
        return

    not_configured = "-"
    enabled = "Enabled"
    disabled = "Disabled"

    rows: list[str | Text | tuple[str | Text, str | Text]] = []

    if function_proto.is_class:
        rows.append(("Cls Name:", function_proto.function_name))
    else:
        rows.append(("Function Name:", function_proto.function_name))

    rows.append(("Function ID:", f.object_id))
    rows.append(("App ID:", handle_metadata.app_id))
    rows.append(("Image ID:", str(info.image_info.image_id)))

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
            "  Scaledown Window:",
            not_configured
            if not autoscaler_settings.scaledown_window
            else f"{autoscaler_settings.scaledown_window} seconds",
        )
    )

    rows.append("Execution:")
    rows.append(("  Timeout:", f"{info.timeout} seconds"))
    rows.append(("  Max Retries:", not_configured if info.max_retries is None else str(info.max_retries)))

    if info.concurrency_info:
        rows.append("Concurrency:")
        rows.append(("  Max Inputs:", str(info.concurrency_info.max_inputs)))
        rows.append(
            (
                "  Target Inputs:",
                not_configured
                if info.concurrency_info.target_inputs is None
                else str(info.concurrency_info.target_inputs),
            )
        )

    if info.batching_info:
        rows.append("Batching:")
        rows.append(("  Max Batch Size:", str(info.batching_info.max_batch_size)))
        rows.append(("  Wait Time:", f"{info.batching_info.wait_ms} milliseconds"))

    # --- Scheduling ---
    rows.append("Scheduling:")
    rows.append(("  Compute Region(s):", not_configured if not info.regions else " | ".join(info.regions)))
    rows.append(("  Nonpreemptible Capacity:", not_configured if not info.nonpreemptible else enabled))
    rows.append(("  Cloud Provider:", not_configured if info.cloud is None else (info.cloud)))
    rows.append(("  Routing Region:", not_configured if not info.routing_region else (info.routing_region)))

    if info.cluster_info:
        rows.append("Clustering:")
        rows.append(("  Cluster Size:", str(info.cluster_info.size)))
        rows.append(("  RDMA:", enabled if info.cluster_info.rdma else disabled))
        if info.cluster_info.fabric_size is not None:
            rows.append(("  Fabric Size:", str(info.cluster_info.fabric_size)))

    if info.schedule is not None:
        rows.append(("Schedule:", info.schedule))

    rows.append("Security:")
    rows.append(("  Outbound Networking:", disabled if info.block_network else enabled))
    rows.append(("  Modal API Access:", disabled if info.restrict_modal_access else enabled))
    rows.append(("  Container Reuse:", disabled if info.single_use_containers else enabled))

    # --- Web Functions ---
    if info.web_info is not None:
        rows.append("Web Info:")

        if info.web_info.web_url:
            url_str = "  URL:"
            if info.web_info.method:
                url_str = f"  URL ({info.web_info.method}):"

            rows.append((url_str, info.web_info.web_url))

        rows.append(("  Authentication:", disabled if info.web_info.unauthenticated else enabled))

    # --- Cls ---
    if info.method_names is not None:
        assert info.method_details is not None

        rows.append("Methods:")

        for method_name in info.method_names:
            web_info = info.method_details.get(method_name)
            if not web_info:
                rows.append(f"  {method_name}")
                continue

            rows.append(f"  {method_name}:")

            if web_info.web_url:
                url_str = "    URL:"
                if web_info.method:
                    url_str = f"    URL ({web_info.method}):"

                rows.append((url_str, web_info.web_url))

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
