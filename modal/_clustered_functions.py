# Copyright Modal Labs 2024
import os
import socket
from dataclasses import dataclass

from modal._utils.async_utils import synchronize_api
from modal._utils.grpc_utils import retry_transient_errors
from modal.client import _Client
from modal_proto import api_pb2


@dataclass
class ClusterInfo:
    rank: int
    world_size: int
    container_ips: list[str]


cluster_info: ClusterInfo | None = None


def get_cluster_info() -> ClusterInfo:
    if cluster_info is None:
        raise Exception("Cluster info not initialized; please ensure that the function is a clustered function")
    return cluster_info


async def _initialize_clustered_function(client: _Client, task_id: str):
    global cluster_info

    def get_i6pn():
        """Returns the ipv6 address assigned to this container."""
        return socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0]

    hostname = socket.gethostname()
    container_ip = get_i6pn()

    # nccl's default host ID is $(hostname)$(cat /proc/sys/kernel/random/boot_id).
    # on runc, if two i6pn-linked containers get scheduled on the same worker,
    # their boot ID and hostname will both be identical, causing nccl to break.
    # As a workaround, we can explicitly specify a unique host ID here.
    # See MOD-4067.
    os.environ["NCCL_HOSTID"] = f"{hostname}{container_ip}"

    resp: api_pb2.ClusterHelloResponse = await retry_transient_errors(
        client.stub.ClusterHello,
        api_pb2.ClusterHelloRequest(
            task_id=task_id,
            container_ip=container_ip,
        ),
    )

    cluster_info = ClusterInfo(
        rank=resp.cluster_rank,
        world_size=len(resp.container_ips),
        container_ips=resp.container_ips,
    )


initialize_clustered_function = synchronize_api(_initialize_clustered_function)
