# Copyright Modal Labs 2026
"""Public data types returned by Modal APIs."""

import enum
from dataclasses import FrozenInstanceError, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Optional, TypedDict

from modal_proto import api_pb2


class InputStatus(enum.IntEnum):
    """Enum representing status of a function input."""

    PENDING = 0
    SUCCESS = api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    FAILURE = api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    INIT_FAILURE = api_pb2.GenericResult.GENERIC_STATUS_INIT_FAILURE
    TERMINATED = api_pb2.GenericResult.GENERIC_STATUS_TERMINATED
    TIMEOUT = api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT

    @classmethod
    def _missing_(cls, value):
        return cls.PENDING


class FileEntryType(enum.IntEnum):
    """Type of a file entry listed from a Modal volume."""

    UNSPECIFIED = 0
    FILE = 1
    DIRECTORY = 2
    SYMLINK = 3
    FIFO = 4
    SOCKET = 5


class FileType(enum.Enum):
    """Type of a filesystem entry."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


class FileWatchEventType(enum.Enum):
    """Type of a filesystem watch event reported by `Sandbox.filesystem.watch()`."""

    Unknown = "Unknown"
    Access = "Access"
    Create = "Create"
    Modify = "Modify"
    Remove = "Remove"


@dataclass
class DictInfo:
    """Information about a Dict object."""

    # This dataclass should be limited to information that is unchanging over the lifetime of the Dict,
    # since it is transmitted from the server when the object is hydrated and could be stale when accessed.

    name: str | None
    created_at: datetime
    created_by: str | None


@dataclass
class QueueInfo:
    """Information about a Queue object."""

    # This dataclass should be limited to information that is unchanging over the lifetime of the Queue,
    # since it is transmitted from the server when the object is hydrated and could be stale when accessed.

    name: str | None
    created_at: datetime
    created_by: str | None


@dataclass
class SecretInfo:
    """Information about a Secret object."""

    # This dataclass should be limited to information that is unchanging over the lifetime of the Secret,
    # since it is transmitted from the server when the object is hydrated and could be stale when accessed.

    name: str | None
    created_at: datetime
    created_by: str | None


class VolumeCreateOptions(TypedDict, total=False):
    """Options used when creating a Volume."""

    experimental_options: dict[str, Any]


@dataclass
class VolumeInfo:
    """Information about a Volume object."""

    # This dataclass should be limited to information that is unchanging over the lifetime of the Volume,
    # since it is transmitted from the server when the object is hydrated and could be stale when accessed.

    name: str | None
    created_at: datetime
    created_by: str | None


# Wrapper type for api_pb2.FunctionStats
@dataclass(frozen=True)
class FunctionStats:
    """Simple data structure storing stats for a running function."""

    backlog: int
    num_total_runners: int
    num_running_inputs: int
    input_headroom: int


@dataclass
class InputInfo:
    """Simple data structure storing information about a function input."""

    input_id: str
    function_call_id: str
    task_id: str
    status: InputStatus
    function_name: str
    module_name: str
    children: list["InputInfo"]


@dataclass(frozen=True)
class FileEntry:
    """A file or directory entry listed from a Modal volume."""

    path: str
    type: FileEntryType
    mtime: int
    size: int

    @classmethod
    def _from_proto(cls, proto: api_pb2.FileEntry) -> "FileEntry":
        return cls(
            path=proto.path,
            type=FileEntryType(proto.type),
            mtime=proto.mtime,
            size=proto.size,
        )


@dataclass(frozen=True)
class FileInfo:
    """Metadata for a file or directory entry in a Sandbox."""

    name: str
    path: str
    type: FileType
    size: int
    mode: int
    permissions: str
    owner: str
    group: str
    modified_time: float
    symlink_target: str | None

    def is_file(self) -> bool:
        """Return `True` if this entry is a regular file."""
        return self.type == FileType.FILE

    def is_dir(self) -> bool:
        """Return `True` if this entry is a directory."""
        return self.type == FileType.DIRECTORY

    def is_symlink(self) -> bool:
        """Return `True` if this entry is a symbolic link."""
        return self.type == FileType.SYMLINK


@dataclass
class FileWatchEvent:
    """A filesystem change event reported by `Sandbox.filesystem.watch()`.

    `paths` contains the absolute path(s) affected by the event. For most
    event types it holds a single entry. Rename operations are reported as
    `Modify` events: when both the source and destination fall within the
    watched scope, `paths` holds `[source, destination]`; when only one
    side of the rename is visible, `paths` holds that single path.
    """

    paths: list[str]
    type: FileWatchEventType


@dataclass(frozen=True)
class SandboxConnectCredentials:
    """Simple data structure storing credentials for making HTTP connections to a sandbox."""

    url: str
    token: str


@dataclass(frozen=True)
class WorkspaceMemberInfo:
    """Metadata about a Workspace member."""

    name: str
    email: str
    user_id: str
    role: str
    joined_at: datetime
    last_active_at: Optional[datetime]  # None if the member has never been active


@dataclass(frozen=True)
class TokenData:
    """A token ID / secret pair."""

    token_id: str
    token_secret: str


@dataclass(frozen=True)
class ProxyTokenInfo:
    """Metadata about a proxy token, not including the token secret."""

    token_id: str
    created_at: datetime
    scoped: bool


@dataclass(frozen=True)
class WorkspaceSettings:
    """Current settings for the workspace."""

    default_environment: str
    image_builder_version: str


@dataclass(slots=True, frozen=True)
class BillingReportItem:
    """Costs generated by a specific object during a specific time interval."""

    object_id: str
    description: str
    environment_name: str
    interval_start: datetime
    cost: Decimal
    cost_by_resource: dict[str, Decimal]
    tags: dict[str, str]

    def __getitem__(self, key: str) -> Any:
        """mdmd:hidden"""
        if key not in self.__slots__:
            raise KeyError(key)

        return getattr(self, key)

    def __setitem__(self, key: str, _: Any):
        """mdmd:hidden"""
        raise FrozenInstanceError(f"cannot assign to field {key!r}")

    def keys(self) -> Iterable[str]:
        """mdmd:hidden"""
        yield from self.__slots__

    def values(self) -> Iterable[Any]:
        """mdmd:hidden"""
        for k in self.__slots__:
            yield getattr(self, k)

    def items(self) -> Iterable[tuple[str, Any]]:
        """mdmd:hidden"""
        for k in self.__slots__:
            yield k, getattr(self, k)

    @classmethod
    def _from_proto(cls, pb_item: "api_pb2.WorkspaceBillingReportItem") -> "BillingReportItem":
        return cls(
            object_id=pb_item.object_id,
            description=pb_item.description,
            environment_name=pb_item.environment_name,
            interval_start=pb_item.interval.ToDatetime().replace(tzinfo=timezone.utc),
            cost=Decimal(pb_item.cost),
            cost_by_resource={k: Decimal(v) for k, v in pb_item.cost_by_resource.items()},
            tags=dict(pb_item.tags),
        )
