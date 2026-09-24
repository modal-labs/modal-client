# Copyright Modal Labs 2025

from typing import Protocol

from modal._cluster import get_current_cluster_context
from modal_proto import api_pb2

from .._runtime.task_lifecycle_manager import UserException
from ..config import logger
from ..exception import InvalidError
from ..experimental.flash_server import server_forward
from .flash import flash_forward


class _FlashLifecycleManager(Protocol):
    def stop(self) -> None: ...

    def close(self) -> None: ...


class _FlashContainerEntry:
    """
    A class that manages the lifecycle of Flash manager for Flash containers.

    It is intentional that stop() runs before exit handlers and close(): new requests stop arriving
    before exit handlers run, and in-flight requests keep working until the exit grace period has
    elapsed. For server containers both calls are no-ops.
    """

    flash_manager: _FlashLifecycleManager | None

    def __init__(self, http_config: api_pb2.HTTPConfig, is_server: bool = False):
        self.http_config: api_pb2.HTTPConfig = http_config
        self.flash_manager = None
        self.is_server = is_server

    def enter(self):
        if self.http_config != api_pb2.HTTPConfig():
            try:
                if self.is_server:
                    self.flash_manager = server_forward()
                else:
                    try:
                        rank = get_current_cluster_context().rank
                        if rank != 0:
                            return
                    except InvalidError:
                        pass

                    self.flash_manager = flash_forward(
                        self.http_config.port,
                        startup_timeout=self.http_config.startup_timeout,
                        exit_grace_period=self.http_config.exit_grace_period,
                        h2_enabled=self.http_config.h2_enabled,
                    )
            except Exception as e:
                logger.warning(f"[Modal Flash] Startup failed: {e}")
                raise UserException()

    def stop(self):
        if self.flash_manager:
            self.flash_manager.stop()

    def close(self):
        if self.flash_manager:
            self.flash_manager.close()
