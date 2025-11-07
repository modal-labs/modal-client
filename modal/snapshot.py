# Copyright Modal Labs 2024
from typing import Optional

from modal_proto import api_pb2

from ._load_context import LoadContext
from ._object import _Object
from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from .client import _Client


class _SandboxSnapshot(_Object, type_prefix="sn"):
    """
    > Sandbox memory snapshots are in **early preview**.

    A `SandboxSnapshot` object lets you interact with a stored Sandbox snapshot that was created by calling
    `._experimental_snapshot()` on a Sandbox instance. This includes both the filesystem and memory state of
    the original Sandbox at the time the snapshot was taken.
    """

    @staticmethod
    async def from_id(sandbox_snapshot_id: str, client: Optional[_Client] = None):
        """
        Construct a `SandboxSnapshot` object from a sandbox snapshot ID.
        """
        # TODO: remove this - from_id constructor should not do io:
        client = client or await _Client.from_env()

        async def _load(
            self: _SandboxSnapshot, resolver: Resolver, load_context: LoadContext, existing_object_id: Optional[str]
        ):
            await load_context.client.stub.SandboxSnapshotGet(
                api_pb2.SandboxSnapshotGetRequest(snapshot_id=sandbox_snapshot_id)
            )

        rep = "SandboxSnapshot()"
        obj = _SandboxSnapshot._from_loader(_load, rep, load_context_overrides=LoadContext(client=client))
        # TODO: should this be a _Object._new_hydrated instead?
        obj._hydrate(sandbox_snapshot_id, client, None)

        return obj


SandboxSnapshot = synchronize_api(_SandboxSnapshot)
