# Copyright Modal Labs 2026
import os
import socket
from dataclasses import dataclass
from typing import Literal, cast

from google.protobuf.message import Message

from modal_proto import api_pb2

from ._load_context import LoadContext
from ._object import _Object, live_method
from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from .client import _Client
from .exception import InvalidError


@dataclass
class ClusterContext:
    """Cluster metadata and the executing container's rank, initialized at startup."""

    rank: int
    cluster_id: str
    container_ips: list[str]
    container_ipv4_ips: list[str]
    fabric_ids: list[str]


current_cluster_context: ClusterContext | None = None


def get_current_cluster_context() -> ClusterContext:
    """Return the executing container's initialized cluster context without an RPC."""
    if current_cluster_context is None:
        raise InvalidError(
            "Cluster context is not initialized. This operation requires execution "
            "inside a clustered Function or Server."
        )
    return current_cluster_context


class _Cluster(_Object, type_prefix="cu"):
    """A group of containers scheduled together for a clustered Function or Server.

    Use `Cluster.from_context()` inside a cluster, or `Cluster.from_id()` to
    inspect a cluster remotely. Containers are ordered by cluster rank.
    """

    _metadata: api_pb2.ClusterStats | None = None

    def _hydrate_metadata(self, metadata: Message | None):
        if metadata is not None:
            assert isinstance(metadata, api_pb2.ClusterStats)
            self._metadata = metadata

    def _get_metadata(self) -> api_pb2.ClusterStats:
        assert self._metadata is not None
        return self._metadata

    @staticmethod
    def from_context() -> "_Cluster":
        """Reference a Cluster from within one of its containers.

        Raises `InvalidError` outside an initialized clustered execution.
        """
        context = get_current_cluster_context()
        if not context.cluster_id:
            raise InvalidError("The executing container does not have an initialized cluster ID.")
        return _Cluster.from_id(context.cluster_id)

    @staticmethod
    def from_id(cluster_id: str, *, client: _Client | None = None) -> "_Cluster":
        """Reference a Cluster by its ID.

        Args:
            cluster_id: ID of the cluster.
            client: Modal client to use; defaults to `Client.from_env()` when omitted.

        Examples:
            ```python notest
            cluster = modal.Cluster.from_id("cu-123")
            ```
        """

        async def _load(self: _Cluster, resolver: Resolver, context: LoadContext, existing_object_id: str | None):
            response = await context.client._stub.ClusterGet(api_pb2.ClusterGetRequest(cluster_id=cluster_id))
            self._hydrate(cluster_id, context.client, response.cluster)

        obj = _Cluster._from_loader(
            _load,
            f"modal.Cluster.from_id({cluster_id!r})",
            hydrate_lazily=True,
            load_context_overrides=LoadContext(client=client),
        )
        obj._object_id = cluster_id
        return obj

    @property
    def object_id(self) -> str:
        """The cluster's unique `cu-` object ID."""
        return super().object_id

    @live_method
    async def container_ids(self) -> list[str]:
        """Return container IDs ordered by cluster rank."""
        return list(self._get_metadata().task_ids)

    async def container_ips(self, family: Literal["ipv4", "ipv6"] = "ipv6") -> list[str]:
        """Return container IP addresses ordered by cluster rank.

        Returns IPv6 addresses by default; pass `family="ipv4"` for IPv4.
        These addresses are for intra-cluster communication.

        Must be called from a container in this cluster; otherwise raises `InvalidError`.
        """
        if family not in ("ipv4", "ipv6"):
            raise InvalidError("family must be 'ipv4' or 'ipv6'.")
        context = get_current_cluster_context()
        if context.cluster_id != self.object_id:
            raise InvalidError("container_ips() must be called from a container in the referenced cluster.")
        return list(context.container_ipv4_ips if family == "ipv4" else context.container_ips)

    async def container_rank(self, container_id: str | None = None) -> int:
        """Return a container's rank within this cluster.

        With no argument, return the executing container's rank without a network
        request. Raises `InvalidError` if it is not a member of this cluster.

        With an explicit container ID, look up its rank in the cluster's membership.
        Raises `InvalidError` for nonmembers.
        """
        if container_id is None:
            context = get_current_cluster_context()
            if context.cluster_id != self.object_id:
                raise InvalidError("The executing container is not a member of the referenced cluster.")
            return context.rank

        container_ids = await self.container_ids()
        try:
            return container_ids.index(container_id)
        except ValueError:
            raise InvalidError(f"Container {container_id!r} is not a member of cluster {self.object_id!r}.") from None


async def _initialize_clustered_function(client: _Client, task_id: str):
    global current_cluster_context

    def get_i6pn() -> str:
        """Returns the ipv6 address assigned to this container."""
        # An AF_INET6 result always carries a (host, port, flowinfo, scope_id) sockaddr,
        # but getaddrinfo is typed for every address family, some of which lead with an int
        return cast(str, socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0])

    hostname = socket.gethostname()
    container_ip = get_i6pn()

    # nccl's default host ID is $(hostname)$(cat /proc/sys/kernel/random/boot_id).
    # on runc, if two i6pn-linked containers get scheduled on the same worker,
    # their boot ID and hostname will both be identical, causing nccl to break.
    # As a workaround, we can explicitly specify a unique host ID here.
    # See MOD-4067.
    os.environ["NCCL_HOSTID"] = f"{hostname}{container_ip}"

    # We found these settings to work well in most cases. You may be able to achieve
    # better performance by tuning these settings.
    if os.environ["MODAL_CLOUD_PROVIDER"] in ("CLOUD_PROVIDER_GCP", "CLOUD_PROVIDER_OCI"):
        os.environ["NCCL_SOCKET_NTHREADS"] = "4"
        os.environ["NCCL_NSOCKS_PERTHREAD"] = "1"
    elif os.environ["MODAL_CLOUD_PROVIDER"] == "CLOUD_PROVIDER_AWS":
        os.environ["NCCL_SOCKET_NTHREADS"] = "2"
        os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"
    else:
        os.environ["NCCL_SOCKET_NTHREADS"] = "1"
        os.environ["NCCL_NSOCKS_PERTHREAD"] = "1"

    resp = await client._stub.TaskClusterHello(
        api_pb2.TaskClusterHelloRequest(
            task_id=task_id,
            container_ip=container_ip,
        ),
    )
    current_cluster_context = ClusterContext(
        rank=resp.cluster_rank,
        cluster_id=resp.cluster_id,
        container_ips=list(resp.container_ips),
        container_ipv4_ips=list(resp.container_ipv4_ips),
        fabric_ids=list(resp.fabric_ids),
    )


initialize_clustered_function = synchronize_api(_initialize_clustered_function)
