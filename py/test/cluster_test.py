# Copyright Modal Labs 2026
import pytest

from grpclib import GRPCError, Status

import modal
from modal import _cluster
from modal._serialization import deserialize, serialize
from modal.exception import InvalidError, NotFoundError
from modal_proto import api_pb2


@pytest.fixture
def cluster_context(monkeypatch, set_env_client):
    info = _cluster.ClusterContext(
        rank=1,
        cluster_id="cu-current",
        container_ips=["fd00::1", "fd00::2"],
        container_ipv4_ips=["10.100.0.1", "10.100.0.2"],
        fabric_ids=["0", "0"],
    )
    monkeypatch.setattr(_cluster, "current_cluster_context", info)
    return info


def cluster_response(*, container_ids=None, started_at=100):
    return api_pb2.ClusterGetResponse(
        cluster=api_pb2.ClusterStats(
            cluster_id="cu-current",
            task_ids=container_ids if container_ids is not None else ["ta-first", "ta-second"],
            started_at=started_at,
        )
    )


def test_context_access_is_local(cluster_context, servicer):
    assert isinstance(cluster_context, modal.experimental.ClusterInfo)
    assert modal.experimental.get_cluster_info() is cluster_context
    with servicer.intercept() as ctx:
        cluster = modal.Cluster.from_context()
        assert cluster.object_id == "cu-current"
        assert cluster.container_rank() == 1
        assert cluster.container_ips() == ["fd00::1", "fd00::2"]
        assert cluster.container_ips(family="ipv4") == ["10.100.0.1", "10.100.0.2"]
        # Returned collections must not let user code corrupt the runtime context.
        cluster.container_ips().clear()
        assert len(cluster.container_ips()) == 2
        assert ctx.calls == []


def test_public_cluster_namespace():
    assert {name for name in vars(modal.cluster) if not name.startswith("_")} == {"Cluster"}
    assert modal.Cluster is modal.cluster.Cluster
    assert modal.Cluster.__module__ == "modal.cluster"
    assert "Cluster" in modal.__all__
    assert not hasattr(modal, "ClusterInfo")
    assert not hasattr(modal.types, "ClusterInfo")
    assert not hasattr(modal.Cluster, "info")
    for module in (modal, modal.cluster, modal.types, modal.experimental):
        assert not hasattr(module, "ClusterContext")
        assert not hasattr(module, "get_current_cluster_context")


@pytest.fixture
def membership(servicer):
    container_ids = ["ta-first", "ta-second"]

    async def respond(servicer, stream):
        await stream.send_message(cluster_response(container_ids=container_ids))

    with servicer.intercept() as ctx:
        ctx.set_responder("ClusterGet", respond)
        yield ctx, container_ids


@pytest.mark.parametrize("first_operation", ["container_ids", "rank"])
def test_context_membership_is_lazy(cluster_context, membership, first_operation):
    ctx, container_ids = membership
    cluster = modal.Cluster.from_context()
    assert cluster.object_id == "cu-current"
    assert ctx.calls == []
    if first_operation == "container_ids":
        assert cluster.container_ids() == container_ids
    else:
        assert cluster.container_rank("ta-second") == 1
    assert ctx.pop_request("ClusterGet").cluster_id == "cu-current"
    assert ctx.calls == []
    cluster.container_ids().clear()
    assert cluster.container_ids() == container_ids
    assert cluster.container_rank("ta-second") == 1
    assert ctx.calls == []


def test_context_required(monkeypatch):
    monkeypatch.setattr(_cluster, "current_cluster_context", None)
    with pytest.raises(InvalidError):
        modal.Cluster.from_context()
    cluster = modal.Cluster.from_id("cu-remote")
    with pytest.raises(InvalidError):
        cluster.container_ips()
    with pytest.raises(InvalidError):
        cluster.container_rank()


def test_context_must_match_referenced_cluster(cluster_context):
    cluster = modal.Cluster.from_id("cu-other")
    with pytest.raises(InvalidError, match="referenced cluster"):
        cluster.container_ips()
    with pytest.raises(InvalidError, match="referenced cluster"):
        cluster.container_rank()
    # Constructor choice does not affect access to the executing cluster.
    assert modal.Cluster.from_id("cu-current").container_rank() == 1


def test_address_family_validation(cluster_context):
    with pytest.raises(InvalidError, match="family"):
        modal.Cluster.from_context().container_ips("ipv5")  # type: ignore


@pytest.mark.parametrize("cluster_id", ["", "ta-123", "cu-"])
def test_invalid_cluster_id_is_checked_on_use(cluster_id, servicer, client):
    async def missing_cluster(servicer, stream):
        raise GRPCError(Status.NOT_FOUND, "Cluster not found")

    with servicer.intercept() as ctx:
        ctx.set_responder("ClusterGet", missing_cluster)
        cluster = modal.Cluster.from_id(cluster_id, client=client)
        assert ctx.calls == []
        with pytest.raises(NotFoundError):
            cluster.container_ids()
        assert ctx.pop_request("ClusterGet").cluster_id == cluster_id


def test_remote_membership_and_rank(membership, client, monkeypatch):
    monkeypatch.setattr(_cluster, "current_cluster_context", None)
    ctx, container_ids = membership
    cluster = modal.Cluster.from_id("cu-current", client=client)
    assert cluster.object_id == "cu-current"
    assert ctx.calls == []
    cluster.hydrate()
    assert ctx.pop_request("ClusterGet").cluster_id == "cu-current"
    assert ctx.calls == []
    assert cluster.container_ids() == container_ids
    cluster.container_ids().clear()
    assert cluster.container_ids() == container_ids

    assert cluster.container_rank("ta-second") == 1
    with pytest.raises(InvalidError, match="not a member"):
        cluster.container_rank("ta-outsider")
    assert ctx.calls == []


def test_hydration_checks_existence(servicer, client):
    async def missing_cluster(servicer, stream):
        raise GRPCError(Status.NOT_FOUND, "Cluster not found")

    cluster = modal.Cluster.from_id("cu-current", client=client)
    with servicer.intercept() as ctx:
        ctx.set_responder("ClusterGet", missing_cluster)
        with pytest.raises(NotFoundError):
            cluster.hydrate()

    with servicer.intercept() as ctx:
        ctx.add_response("ClusterGet", cluster_response())
        cluster.hydrate()


def test_hydrated_cluster_serialization(membership, client):
    ctx, container_ids = membership
    cluster = modal.Cluster.from_id("cu-current", client=client).hydrate()
    restored = deserialize(serialize(cluster), client)
    ctx.calls.clear()
    assert isinstance(restored, modal.Cluster)
    assert restored.container_ids() == container_ids
    assert restored.container_rank("ta-second") == 1
    assert ctx.calls == []


@pytest.mark.parametrize("started_at", [0, 100])
def test_membership_does_not_require_start_timestamp(servicer, client, started_at):
    with servicer.intercept() as ctx:
        ctx.add_response("ClusterGet", cluster_response(started_at=started_at))
        cluster = modal.Cluster.from_id("cu-current", client=client)
        assert cluster.container_ids() == ["ta-first", "ta-second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("from_context", [False, True])
async def test_async_access(membership, client, cluster_context, from_context):
    ctx, container_ids = membership
    cluster = modal.Cluster.from_context() if from_context else modal.Cluster.from_id("cu-current", client=client)
    assert await cluster.container_rank.aio() == 1
    assert await cluster.container_ips.aio() == ["fd00::1", "fd00::2"]
    assert await cluster.container_ips.aio(family="ipv4") == ["10.100.0.1", "10.100.0.2"]
    (await cluster.container_ips.aio()).clear()
    assert await cluster.container_ips.aio() == ["fd00::1", "fd00::2"]
    assert ctx.calls == []
    assert await cluster.container_ids.aio() == container_ids
    assert ctx.pop_request("ClusterGet").cluster_id == "cu-current"
    assert await cluster.container_ids.aio() == container_ids
    assert await cluster.container_rank.aio("ta-second") == 1
    with pytest.raises(InvalidError, match="not a member"):
        await cluster.container_rank.aio("ta-outsider")
    assert ctx.calls == []


@pytest.mark.parametrize("world_size", [1, 2])
def test_cluster_initialization(servicer, client, set_env_client, monkeypatch, world_size):
    monkeypatch.setattr(_cluster, "current_cluster_context", None)
    monkeypatch.setenv("MODAL_CLOUD_PROVIDER", "CLOUD_PROVIDER_AWS")
    monkeypatch.setattr("modal._cluster.socket.getaddrinfo", lambda *args: [(0, 0, 0, "", ("fd00::1", 0, 0, 0))])
    container_ids = [f"ta-{rank}" for rank in range(world_size)]
    container_ips = [f"fd00::{rank + 1}" for rank in range(world_size)]
    ipv4_ips = [f"10.100.0.{rank + 1}" for rank in range(world_size)]
    with servicer.intercept() as ctx:
        ctx.add_response(
            "TaskClusterHello",
            api_pb2.TaskClusterHelloResponse(
                cluster_id="cu-current",
                cluster_rank=0,
                container_ips=container_ips,
                container_ipv4_ips=ipv4_ips,
                fabric_ids=["0"] * world_size,
            ),
        )
        _cluster.initialize_clustered_function(client, "ta-0")
        assert ctx.pop_request("TaskClusterHello").task_id == "ta-0"

    with servicer.intercept() as ctx:
        cluster = modal.Cluster.from_context()
        assert cluster.object_id == "cu-current"
        assert cluster.container_rank() == 0
        assert cluster.container_ips() == container_ips
        assert cluster.container_ips("ipv4") == ipv4_ips
        context = modal.experimental.get_cluster_info()
        assert context.container_ipv4_ips == ipv4_ips
        assert context.fabric_ids == ["0"] * world_size
        assert modal.experimental.get_fabric_peers() == list(range(world_size))
        assert ctx.calls == []
        ctx.add_response("ClusterGet", cluster_response(container_ids=container_ids))
        assert cluster.container_ids() == container_ids
        assert ctx.pop_request("ClusterGet").cluster_id == "cu-current"
        assert cluster.container_rank(container_ids[-1]) == world_size - 1
        assert ctx.calls == []
