# Copyright Modal Labs 2025
import os

from modal_proto import api_pb2

from .._utils.async_utils import synchronize_api, synchronizer
from ..client import _Client
from ..config import logger


class _ServerManager:
    def __init__(
        self,
        client: _Client,
    ):
        self.client = client
        self.task_id = os.environ["MODAL_TASK_ID"]

    async def _start(self):
        await self.client._stub.ContainerServerLifecycleReady(api_pb2.ContainerServerLifecycleReadyRequest())
        logger.warning(f"[Modal Server] Ready for task {self.task_id}.")

    async def stop(self):
        pass

    async def close(self):
        logger.warning(f"[Modal Server] Drained for task {self.task_id}.")


ServerManager = synchronize_api(_ServerManager, target_module=__name__)


@synchronizer.create_blocking
async def server_forward() -> _ServerManager:
    client = await _Client.from_env()

    manager = _ServerManager(client)
    await manager._start()
    return manager
