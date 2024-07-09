# Copyright Modal Labs 2023
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, BinaryIO, List, Optional, Tuple, Type, Union

from synchronicity.async_wrap import asynccontextmanager

from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from .client import _Client
from .exception import deprecation_error
from .object import (
    EPHEMERAL_OBJECT_HEARTBEAT_SLEEP,
    _Object,
    live_method,
    live_method_gen,
)
from .volume import FileEntry

NETWORK_FILE_SYSTEM_PUT_FILE_CLIENT_TIMEOUT = (
    10 * 60
)  # 10 min max for transferring files (does not include upload time to s3)


def network_file_system_mount_protos(
    validated_network_file_systems: List[Tuple[str, "_NetworkFileSystem"]],
    allow_cross_region_volumes: bool,
) -> List[api_pb2.SharedVolumeMount]:
    """`modal.network_file_system.network_file_system_mount_protos` is deprecated.

    Please use `modal.volume.volume_nfs_mount_protos instead.
    """
    deprecation_error((2024, 7, 9), network_file_system_mount_protos.__doc__)


class _NetworkFileSystem(_Object, type_prefix="sv"):
    @staticmethod
    def new(cloud: Optional[str] = None):
        """`NetworkFileSystem.new` is deprecated.

        Please use `Volume.from_name(nfs=True)` (for persisted) or `Volume.ephemeral(nfs=True)`
        (for ephemeral) network filesystems instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @staticmethod
    def from_name(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> "_NetworkFileSystem":
        """`NetworkFileSystem.from_name` is deprecated.

        Please use `Volume.from_name(nfs=True)` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @classmethod
    @asynccontextmanager
    async def ephemeral(
        cls: Type["_NetworkFileSystem"],
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        _heartbeat_sleep: float = EPHEMERAL_OBJECT_HEARTBEAT_SLEEP,
    ) -> AsyncIterator["_NetworkFileSystem"]:
        """`NetworkFileSystem.ephemeral` is deprecated.

        Please use `Volume.ephemeral(nfs=True)` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @staticmethod
    def persisted(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        cloud: Optional[str] = None,
    ):
        """`NetworkFileSystem.persisted("my-volume")` is deprecated.

        Please use `Volume.from_name(name, create_if_missing=True, nfs=True)` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    def persist(
        self,
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        cloud: Optional[str] = None,
    ):
        """`NetworkFileSystem().persist("my-volume")` is deprecated.

        Please use `Volume.from_name(name, create_if_missing=True, nfs=True)` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @staticmethod
    async def lookup(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> "_NetworkFileSystem":
        """`NetworkFileSystem.lookup` is deprecated.

        Please use `Volume.lookup("my-volume", nfs=True)` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @staticmethod
    async def create_deployed(
        deployment_name: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ) -> str:
        """`NetworkFileSystem.create_deployed` is deprecated.

        Please use `Volume.create_deployed("my-deployment", nfs=True)` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @live_method
    async def write_file(self, remote_path: str, fp: BinaryIO) -> int:
        """`NetworkFileSystem().write_file` is deprecated.

        Please use `Volume().write_file` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @live_method_gen
    async def read_file(self, path: str) -> AsyncIterator[bytes]:
        """`NetworkFileSystem().read_file` is deprecated.

        Please use `Volume().read_file` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @live_method_gen
    async def iterdir(self, path: str) -> AsyncIterator[FileEntry]:
        """`NetworkFileSystem().iterdir` is deprecated.

        Please use `Volume().iterdir("/my/path/prefix**", recursive=False)` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @live_method
    async def add_local_file(
        self, local_path: Union[Path, str], remote_path: Optional[Union[str, PurePosixPath, None]] = None
    ):
        """`NetworkFileSystem().add_local_file` is deprecated.

        Please use `Volume.add_local_file` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @live_method
    async def add_local_dir(
        self,
        local_path: Union[Path, str],
        remote_path: Optional[Union[str, PurePosixPath, None]] = None,
    ):
        """`NetworkFileSystem().add_local_dir` is deprecated.

        Please use `Volume().add_local_dir` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @live_method
    async def listdir(self, path: str) -> List[FileEntry]:
        """`NetworkFileSystem().listdir` is deprecated.

        Please use `Volume().listdir("my/path/prefix**", recursive=False)` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)

    @live_method
    async def remove_file(self, path: str, recursive=False):
        """`NetworkFileSystem().remove_file` is deprecated.

        Please use `Volume().remove_file("my/path/prefix**", recursive=False)` instead.
        """
        deprecation_error((2024, 7, 9), NetworkFileSystem.new.__doc__)


NetworkFileSystem = synchronize_api(_NetworkFileSystem)
