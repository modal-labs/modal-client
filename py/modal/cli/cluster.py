# Copyright Modal Labs 2022
import uuid
from typing import Optional, Union

import click
from rich.table import Column
from rich.text import Text

from modal._object import _get_environment_name
from modal._output.pty import get_pty_info
from modal._utils.async_utils import synchronizer
from modal._utils.task_command_router_client import TaskCommandRouterClient
from modal._utils.time_utils import timestamp_to_localized_str
from modal.cli.utils import display_table, env_option, is_tty
from modal.client import _Client
from modal.config import config
from modal.container_process import _ContainerProcess
from modal.environments import ensure_env
from modal.exception import InvalidError
from modal.output import OutputManager
from modal.stream_type import StreamType
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from ._help import ModalGroup

cluster_cli = ModalGroup(name="cluster", help="Manage and connect to running multi-node clusters.")


@cluster_cli.command("list")
@env_option
@click.option("--json", is_flag=True, default=False)
@synchronizer.create_blocking
async def list_(env: Optional[str] = None, json: bool = False):
    """List all clusters that are currently running."""
    env = ensure_env(env)
    client = await _Client.from_env()
    environment_name = _get_environment_name(env)
    res: api_pb2.ClusterListResponse = await client.stub.ClusterList(
        api_pb2.ClusterListRequest(environment_name=environment_name)
    )

    column_names: list[Union[Column, str]] = [
        Column("Cluster ID", min_width=25),
        Column("App ID", min_width=25),
        "Start Time",
        "Nodes",
    ]
    rows: list[list[Union[Text, str]]] = []
    res.clusters.sort(key=lambda c: c.started_at, reverse=True)

    for c in res.clusters:
        rows.append(
            [
                c.cluster_id,
                c.app_id,
                timestamp_to_localized_str(c.started_at, json) if c.started_at else "Pending",
                str(len(c.task_ids)),
            ]
        )

    display_table(column_names, rows, json=json, title=f"Active Multi-node Clusters in environment: {environment_name}")


@cluster_cli.command("shell")
@click.argument("cluster_id")
@click.option("--rank", default=0, help="Rank of the node to shell into")
@synchronizer.create_blocking
async def shell(cluster_id: str, rank: int = 0):
    """Open a shell to a multi-node cluster node."""
    client = await _Client.from_env()
    res: api_pb2.ClusterGetResponse = await client.stub.ClusterGet(api_pb2.ClusterGetRequest(cluster_id=cluster_id))
    if len(res.cluster.task_ids) <= rank:
        raise click.ClickException(f"No node with rank {rank} in cluster {cluster_id}")
    task_id = res.cluster.task_ids[rank]
    is_main = "(main)" if rank == 0 else ""
    OutputManager.get().print(
        f"[green]Opening shell to node {rank} {is_main} of cluster {cluster_id} (container {task_id})[/green]"
    )

    pty = is_tty()

    command_router_client = await TaskCommandRouterClient.try_init(client, task_id)
    if command_router_client is None:
        raise InvalidError(f"Command router access is not available for container {task_id}")

    process_id = str(uuid.uuid4())

    start_req = sr_pb2.TaskExecStartRequest(
        task_id=task_id,
        exec_id=process_id,
        command_args=["/bin/bash"],
        stdout_config=sr_pb2.TaskExecStdoutConfig.TASK_EXEC_STDOUT_CONFIG_PIPE,
        stderr_config=sr_pb2.TaskExecStderrConfig.TASK_EXEC_STDERR_CONFIG_PIPE,
        pty_info=get_pty_info(shell=True) if pty else None,
        runtime_debug=config.get("function_runtime_debug"),
    )
    await command_router_client.exec_start(start_req)

    if pty:
        await _ContainerProcess(process_id, task_id, client, command_router_client=command_router_client).attach()
    else:
        await _ContainerProcess(
            process_id,
            task_id,
            client,
            command_router_client=command_router_client,
            stdout=StreamType.STDOUT,
            stderr=StreamType.STDOUT,
        ).wait()
