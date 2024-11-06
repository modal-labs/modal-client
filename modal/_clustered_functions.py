# Copyright Modal Labs 2024
import os
import socket

from modal._utils.async_utils import synchronize_api
from modal._utils.grpc_utils import retry_transient_errors
from modal.client import _Client
from modal_proto import api_pb2


async def _initialize_clustered_function(client: _Client, task_id: str):
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

    print(f"Initializing clustered function with task ID {task_id} and IP {container_ip}")
    resp: api_pb2.ClusterHelloResponse = await retry_transient_errors(
        client.stub.ClusterHello,
        api_pb2.ClusterHelloRequest(
            task_id=task_id,
            container_ip=container_ip,
        ),
    )
    container_ips = resp.container_ips
    cluster_id = resp.cluster_id
    cluster_rank = resp.cluster_rank
    cluster_size = resp.cluster_size

    print(
        f"Cluster initialized with ID {cluster_id}, rank {cluster_rank}, size {cluster_size}, "
        f"and IPs {container_ips}"
    )


initialize_clustered_function = synchronize_api(_initialize_clustered_function)
