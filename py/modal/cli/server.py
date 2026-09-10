# Copyright Modal Labs 2026
from __future__ import annotations

import click

from modal._environments import ensure_env
from modal._utils.async_utils import synchronizer
from modal.server import Server

from ._help import ModalGroup
from ._logs import _get_app_id_for_function_id, _run_logs_command, _validate_logs_args
from .utils import _is_function_id, _parse_function_or_server_ref, env_option

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

    if _is_function_id(server_ref):
        function_id = server_ref
        app_id = await _get_app_id_for_function_id(function_id)
    else:
        app_name, server_name = _parse_function_or_server_ref(server_ref, "Server")
        server = Server.from_name(app_name, server_name, environment_name=env)
        query_data = await server._get_log_query_data.aio()
        function_id = query_data.source_object_id
        app_id = query_data.app_id

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
