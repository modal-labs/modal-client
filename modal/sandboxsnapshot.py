# Copyright Modal Labs 2024

from typing import Optional

from modal_proto import api_pb2

from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import retry_transient_errors
from .client import _Client
from .object import _Object


class _SandboxSnapshot(_Object, type_prefix="sn"):
    """A `SandboxSnapshot` object lets you interact with a stored Sandbox snapshot that was created by calling
    .snapshot() on a Sandbox instance. This includes both the filesystem and memory state of the original Sandbox at the
    time the snapshot was taken.
    """

    @staticmethod
    async def from_id(sandbox_snapshot_id: str, client: Optional[_Client] = None):
        if client is None:
            client = await _Client.from_env()

        async def _load(self: _SandboxSnapshot, resolver: Resolver, existing_object_id: Optional[str]):
            resp = await retry_transient_errors(
                client.stub.SandboxSnapshotFromId,
                api_pb2.SandboxSnapshotFromIdRequest(sandbox_snapshot_id=sandbox_snapshot_id),
            )
            self._hydrate(resp.sandbox_snapshot_id, resolver.client, resp.metadata)

        rep = "SandboxSnapshot()"
        obj = _SandboxSnapshot._from_loader(_load, rep)

        return obj


SandboxSnapshot = synchronize_api(_SandboxSnapshot)
