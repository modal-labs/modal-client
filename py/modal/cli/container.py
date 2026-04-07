# Copyright Modal Labs 2022
import warnings
from datetime import datetime, timezone
from typing import Optional, Union

import typer
from click import UsageError
from rich.table import Column
from rich.text import Text

from modal._logs import _FETCH_LIMIT, _MAX_FETCH_RANGE, LogsFilters
from modal._object import _get_environment_name
from modal._output.pty import get_pty_info
from modal._utils.async_utils import synchronizer
from modal._utils.time_utils import timestamp_to_localized_str
from modal.cli.app import _DEFAULT_LOGS_TAIL, _SOURCE_OPTIONS, _parse_time_arg
from modal.cli.utils import (
    ENV_OPTION,
    YES_OPTION,
    confirm_or_suggest_yes,
    display_table,
    fetch_app_logs,
    is_tty,
    stream_app_logs,
    tail_app_logs,
)
from modal.client import _Client
from modal.config import config
from modal.container_process import _ContainerProcess
from modal.environments import ensure_env
from modal.exception import InvalidError
from modal.stream_type import StreamType
from modal_proto import api_pb2

container_cli = typer.Typer(name="container", help="Manage and connect to running containers.", no_args_is_help=True)


@container_cli.command("list")
@synchronizer.create_blocking
async def list_(
    app_id: str = typer.Option("", "--app-id", help="List containers running for a specific App."),
    env: Optional[str] = ENV_OPTION,
    json: bool = False,
):
    """List all containers that are currently running."""
    env = ensure_env(env)
    client = await _Client.from_env()
    environment_name = _get_environment_name(env)
    res: api_pb2.TaskListResponse = await client.stub.TaskList(
        api_pb2.TaskListRequest(environment_name=environment_name, app_id=app_id)
    )

    column_names: list[Union[Column, str]] = [
        Column("Container ID", min_width=29),
        Column("App ID", min_width=25),
        "App Name",
        "Start Time",
    ]
    rows: list[list[Union[Text, str]]] = []
    res.tasks.sort(key=lambda task: task.started_at, reverse=True)
    for task_stats in res.tasks:
        rows.append(
            [
                task_stats.task_id,
                task_stats.app_id,
                task_stats.app_description,
                timestamp_to_localized_str(task_stats.started_at, json) if task_stats.started_at else "Pending",
            ]
        )

    display_table(column_names, rows, json=json, title=f"Active Containers in environment: {environment_name}")


@container_cli.command("logs", no_args_is_help=True)
@synchronizer.create_blocking
async def logs(
    container_id: str = typer.Argument(help="Container ID"),
    follow: bool = typer.Option(False, "-f", "--follow", help="Stream log output until Container stops"),
    all_logs: bool = typer.Option(False, "--all", help="Show all logs for the container"),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help=(
            "Start of time range. Accepts ISO 8601 datetime or relative time, e.g. '1d' (1 day ago), '2h', '30m', etc."
        ),
    ),
    until: Optional[str] = typer.Option(
        None,
        "--until",
        help="End of time range; accepts same argument types as --since",
    ),
    tail: Optional[int] = typer.Option(None, "--tail", "-n", help="Show only the last N log entries"),
    search: Optional[str] = typer.Option(None, "--search", help="Filter by search text"),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Filter by source: 'stdout', 'stderr', or 'system'"
    ),
    timestamps: bool = typer.Option(False, "--timestamps", help="Prefix each line with its timestamp"),
):
    """Fetch or stream logs for a specific container.

    By default, this command fetches the last 100 log entries and exits. Use ``-f`` to
    live-stream logs from a running container instead. Fetch and follow are mutually exclusive.

    **Examples:**

    Get recent logs for a container:

    ```
    modal container logs ta-123456
    ```

    Follow (stream) logs from a running container:

    ```
    modal container logs ta-123456 -f
    ```

    Fetch logs from the last 2 hours:

    ```
    modal container logs ta-123456 --since 2h
    ```

    Fetch logs in a specific time range:

    ```
    modal container logs ta-123456 --since 2026-03-01T05:00:00 --until 2026-03-01T08:00:00
    ```

    Fetch the last 1000 entries:

    ```
    modal container logs ta-123456 --tail 1000
    ```

    Fetch all container logs:

    ```
    modal container logs ta-123456 --all
    ```
    """
    task_id, sandbox_id = None, None
    if container_id.startswith("sb-"):
        sandbox_id = container_id
    elif container_id.startswith("ta-"):
        task_id = container_id
    else:
        raise InvalidError(f"Invalid container ID: {container_id}")

    if follow and (since or until or tail):
        raise UsageError("--follow cannot be combined with --since, --until, or --tail.")

    if tail is not None and tail <= 0:
        raise UsageError("--tail value must be positive.")

    if tail is not None and tail > _FETCH_LIMIT:
        raise UsageError(f"--tail value must not exceed {_FETCH_LIMIT}.")

    if all_logs and (since or until or tail):
        raise UsageError("--all cannot be combined with --since, --until, or --tail.")

    if all_logs and follow:
        raise UsageError("--all cannot be combined with --follow.")

    if source is not None:
        if source not in _SOURCE_OPTIONS:
            raise UsageError(f"Invalid source: '{source}'. Must be 'stdout', 'stderr', or 'system'.")
        source_fd = _SOURCE_OPTIONS[source]
    else:
        source_fd = api_pb2.FILE_DESCRIPTOR_UNSPECIFIED

    log_filters = LogsFilters(
        source=source_fd,
        task_id=task_id or "",
        sandbox_id=sandbox_id or "",
        search_text=search or "",
    )

    if follow:
        await stream_app_logs.aio(
            task_id=task_id,
            sandbox_id=sandbox_id,
            show_timestamps=timestamps,
            follow=True,
            filters=log_filters,
        )
    else:
        # Resolve the app_id for the container.
        client = await _Client.from_env()

        if sandbox_id:
            sb_resp = await client.stub.SandboxGetTaskId(api_pb2.SandboxGetTaskIdRequest(sandbox_id=sandbox_id))
            task_id = sb_resp.task_id

        task_info_resp = await client.stub.TaskGetInfo(api_pb2.TaskGetInfoRequest(task_id=task_id))
        app_id = task_info_resp.app_id

        if not task_info_resp.info.started_at:
            # Unlikely race or Modal backend issue, don't treat as a usage exception
            return
        container_started_dt = datetime.fromtimestamp(task_info_resp.info.started_at, timezone.utc)

        now = datetime.now(timezone.utc)
        if all_logs:
            since_dt = container_started_dt
            if task_info_resp.info.finished_at:
                until_dt = datetime.fromtimestamp(task_info_resp.info.finished_at, timezone.utc)
            else:
                until_dt = now
        else:
            since_dt = _parse_time_arg(since, default=container_started_dt)
            if task_info_resp.info.finished_at:
                default_until_dt = datetime.fromtimestamp(task_info_resp.info.finished_at, timezone.utc)
            else:
                default_until_dt = now
            until_dt = _parse_time_arg(until, default=default_until_dt)

        if since is not None and until is not None and since_dt >= until_dt:
            # User provided both --since and --until, but did so incorrectly
            raise UsageError("--since must be before --until.")

        if since is not None and until is None and since_dt >= until_dt:
            # User provided only --since and it is after the container finished
            warnings.warn("--since time is after the Container finished, no logs to fetch.", UserWarning)
            return

        if until is not None and since is None and since_dt >= until_dt:
            # User provided only --until and it is before the Container started
            warnings.warn("--until time is before the Container started, no logs to fetch.", UserWarning)
            return

        if since_dt is not None:
            effective_until = until_dt or now
            if effective_until - since_dt > _MAX_FETCH_RANGE:
                raise UsageError(f"Log fetch time range cannot exceed {_MAX_FETCH_RANGE.days} days.")

        if all_logs or (since and tail is None):
            # Range mode: --since without --tail fetches everything in the range.
            await fetch_app_logs.aio(
                app_id,
                since_dt,
                until_dt or now,
                show_timestamps=timestamps,
                filters=log_filters,
            )
        else:
            # Tail mode: single fetch with limit.
            # --since is a hard floor, --until shifts the anchor.
            effective_tail = tail if tail is not None else _DEFAULT_LOGS_TAIL
            await tail_app_logs.aio(
                app_id,
                effective_tail,
                show_timestamps=timestamps,
                since=since_dt,
                until=until_dt,
                filters=log_filters,
            )


@container_cli.command("exec")
@synchronizer.create_blocking
async def exec(
    pty: Optional[bool] = typer.Option(default=None, help="Run the command using a PTY."),
    container_id: str = typer.Argument(help="Container ID"),
    command: list[str] = typer.Argument(
        help="A command to run inside the container.\n\n"
        "To pass command-line flags or options, add `--` before the start of your commands. "
        "For example: `modal container exec <id> -- /bin/bash -c 'echo hi'`"
    ),
):
    """Execute a command in a container."""

    if pty is None:
        pty = is_tty()

    client = await _Client.from_env()

    req = api_pb2.ContainerExecRequest(
        task_id=container_id,
        command=command,
        pty_info=get_pty_info(shell=True) if pty else None,
        runtime_debug=config.get("function_runtime_debug"),
    )
    res: api_pb2.ContainerExecResponse = await client.stub.ContainerExec(req)

    if pty:
        await _ContainerProcess(res.exec_id, container_id, client).attach()
    else:
        # TODO: redirect stderr to its own stream?
        await _ContainerProcess(
            res.exec_id, container_id, client, stdout=StreamType.STDOUT, stderr=StreamType.STDOUT
        ).wait()


@container_cli.command("stop")
@synchronizer.create_blocking
async def stop(
    container_id: str = typer.Argument(help="Container ID"),
    *,
    yes: bool = YES_OPTION,
):
    """Stop a currently-running container and reassign its in-progress inputs.

    This will send the container a SIGINT signal that Modal will handle.
    """
    client = await _Client.from_env()
    resp = await client.stub.TaskGetInfo(api_pb2.TaskGetInfoRequest(task_id=container_id))
    if resp.info.finished_at:
        raise SystemExit(f"Container '{container_id}' is already stopped.")
    if not yes:
        confirm_or_suggest_yes(f"Are you sure you want to stop container '{container_id}'?")
    request = api_pb2.ContainerStopRequest(task_id=container_id)
    await client.stub.ContainerStop(request)
