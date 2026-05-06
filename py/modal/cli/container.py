# Copyright Modal Labs 2022
import uuid
import warnings
from datetime import datetime, timezone
from typing import Optional, Union

import click
from click import UsageError
from rich.table import Column
from rich.text import Text

from modal._environments import ensure_env
from modal._logs import _FETCH_LIMIT, _MAX_FETCH_RANGE, LogsFilters
from modal._object import _get_environment_name
from modal._output.pty import get_pty_info
from modal._utils.async_utils import synchronizer
from modal._utils.task_command_router_client import TaskCommandRouterClient
from modal._utils.time_utils import timestamp_to_localized_str
from modal.cli.app import _DEFAULT_LOGS_TAIL, _SOURCE_OPTIONS, _parse_time_arg
from modal.cli.utils import (
    confirm_or_suggest_yes,
    display_table,
    env_option,
    fetch_app_logs,
    is_tty,
    stream_app_logs,
    tail_app_logs,
    yes_option,
)
from modal.client import _Client
from modal.config import config
from modal.container_process import _ContainerProcess
from modal.exception import InvalidError
from modal.stream_type import StreamType
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from ._help import ModalGroup

container_cli = ModalGroup(name="container", help="Manage and connect to running containers.")


@container_cli.command("list")
@click.option("--app-id", default="", help="List containers running for a specific App.")
@env_option
@click.option("--json", is_flag=True, default=False)
@synchronizer.create_blocking
async def list_(
    app_id: str = "",
    env: Optional[str] = None,
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
@click.argument("container_id")
@click.option("-f", "--follow", is_flag=True, default=False, help="Stream log output until Container stops")
@click.option("--all", "all_logs", is_flag=True, default=False, help="Show all logs for the container")
@click.option(
    "--since",
    default=None,
    help="Start of time range. Accepts ISO 8601 datetime or relative time, e.g. '1d' (1 day ago), '2h', '30m', etc.",
)
@click.option("--until", default=None, help="End of time range; accepts same argument types as --since")
@click.option("-n", "--tail", default=None, type=int, help="Show only the last N log entries")
@click.option("--search", default=None, help="Filter by search text")
@click.option("-s", "--source", default=None, help="Filter by source: 'stdout', 'stderr', or 'system'")
@click.option("--timestamps", is_flag=True, default=False, help="Prefix each line with its timestamp")
@synchronizer.create_blocking
async def logs(
    container_id: str,
    follow: bool = False,
    all_logs: bool = False,
    since: Optional[str] = None,
    until: Optional[str] = None,
    tail: Optional[int] = None,
    search: Optional[str] = None,
    source: Optional[str] = None,
    timestamps: bool = False,
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


@synchronizer.create_blocking
async def _exec_impl(
    pty: Optional[bool] = None,
    container_id: str = "",
    command: tuple[str, ...] = (),
):
    """Execute a command in a container (implementation)."""

    if pty is None:
        pty = is_tty()

    client = await _Client.from_env()

    command_router_client = await TaskCommandRouterClient.try_init(client, container_id)
    if command_router_client is None:
        raise InvalidError(f"Command router access is not available for container {container_id}")

    process_id = str(uuid.uuid4())

    start_req = sr_pb2.TaskExecStartRequest(
        task_id=container_id,
        exec_id=process_id,
        command_args=command,
        stdout_config=sr_pb2.TaskExecStdoutConfig.TASK_EXEC_STDOUT_CONFIG_PIPE,
        stderr_config=sr_pb2.TaskExecStderrConfig.TASK_EXEC_STDERR_CONFIG_PIPE,
        pty_info=get_pty_info(shell=True) if pty else None,
        runtime_debug=config.get("function_runtime_debug"),
    )
    await command_router_client.exec_start(start_req)

    if pty:
        await _ContainerProcess(process_id, container_id, client, command_router_client=command_router_client).attach()
    else:
        await _ContainerProcess(
            process_id,
            container_id,
            client,
            command_router_client=command_router_client,
            stdout=StreamType.STDOUT,
            stderr=StreamType.STDOUT,
        ).wait()


@container_cli.command("exec")
@click.option("--pty/--no-pty", default=None, help="Run the command using a PTY.")
@click.argument("container_id")
@click.argument("command", nargs=-1, required=True)
def exec(
    pty: Optional[bool] = None,
    container_id: str = "",
    command: tuple[str, ...] = (),
):
    """Execute a command in a container."""
    _exec_impl(pty=pty, container_id=container_id, command=command)


@container_cli.command("stop")
@click.argument("container_id")
@yes_option
@synchronizer.create_blocking
async def stop(container_id: str = "", *, yes: bool = False):
    """Terminate a running container.

    This will send the container a SIGINT signal that Modal will handle.
    Any inputs that are currently running on the container will be cancelled and rescheduled
    on other containers.
    """
    client = await _Client.from_env()
    resp = await client.stub.TaskGetInfo(api_pb2.TaskGetInfoRequest(task_id=container_id))
    if resp.info.finished_at:
        raise SystemExit(f"Container '{container_id}' is already stopped.")
    if not yes:
        confirm_or_suggest_yes(f"Are you sure you want to stop container '{container_id}'?")
    request = api_pb2.ContainerStopRequest(task_id=container_id)
    await client.stub.ContainerStop(request)
