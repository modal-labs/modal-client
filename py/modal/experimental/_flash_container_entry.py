# Copyright Modal Labs 2025


from modal_proto import api_pb2

from .._runtime.task_lifecycle_manager import UserException
from ..config import logger
from .flash_server import ServerManager, server_forward


class _FlashContainerEntry:
    """
    A class that manages the lifecycle of Flash manager for Flash containers.

    It is intentional that stop() runs before exit handlers and close(): new requests stop arriving
    before exit handlers run, and in-flight requests keep working until the exit grace period has
    elapsed. For server containers both calls are no-ops.
    """

    flash_manager: ServerManager | None

    def __init__(self, http_config: api_pb2.HTTPConfig):
        self.http_config: api_pb2.HTTPConfig = http_config
        self.flash_manager = None

    def enter(self):
        if self.http_config != api_pb2.HTTPConfig():
            try:
                self.flash_manager = server_forward()
            except Exception as e:
                logger.warning(f"[Modal Server] Startup failed: {e}")
                raise UserException()

    def close(self):
        if self.flash_manager:
            self.flash_manager.close()
