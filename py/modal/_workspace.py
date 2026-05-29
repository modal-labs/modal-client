# Copyright Modal Labs 2025
import builtins
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from google.protobuf.empty_pb2 import Empty

from modal_proto import api_pb2

from ._load_context import LoadContext
from ._object import _Object
from ._resolver import Resolver
from ._utils.time_utils import timestamp_to_localized_dt
from .client import _Client

MemberRole = Literal["user", "manager", "owner"]


def _member_role_from_proto(proto_value: int) -> MemberRole:
    match proto_value:
        case api_pb2.MEMBER_ROLE_USER:
            return "user"
        case api_pb2.MEMBER_ROLE_MANAGER:
            return "manager"
        case api_pb2.MEMBER_ROLE_OWNER:
            return "owner"
        case _:
            raise ValueError(f"Unknown workspace member role: {proto_value}")


@dataclass(frozen=True)
class WorkspaceMemberInfo:
    """Metadata about a Workspace member."""

    name: str
    email: str
    user_id: str
    role: MemberRole
    joined_at: datetime
    last_active_at: Optional[datetime]  # None if the member has never been active


class _WorkspaceMembersManager:
    """mdmd:namespace
    Namespace with methods for managing the membership of a Workspace.
    """

    def __init__(self, workspace: "_Workspace"):
        """mdmd:hidden"""
        self._workspace = workspace

    async def list(self) -> builtins.list[WorkspaceMemberInfo]:
        """Return the members of the Workspace.

        **Examples:**

        ```python notest
        members = modal.Workspace.from_context().members.list()
        print([m.name for m in members])
        ```
        """
        await self._workspace.hydrate()
        resp = await self._workspace.client.stub.WorkspaceMembersList(Empty())
        return [
            WorkspaceMemberInfo(
                user_id=item.user_id,
                name=item.member_displayname,
                email=item.email,
                role=_member_role_from_proto(item.member_role),
                joined_at=timestamp_to_localized_dt(item.joined_at),
                last_active_at=timestamp_to_localized_dt(item.last_active_at) if item.last_active_at else None,
            )
            for item in sorted(resp.members, key=lambda x: x.member_displayname)
        ]


class _Workspace(_Object, type_prefix="ac"):
    _name: Optional[str] = None

    def __init__(self):
        """mdmd:hidden"""
        raise RuntimeError("`Workspace(...)` constructor is not allowed. Use `Workspace.from_context` instead.")

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def members(self) -> "_WorkspaceMembersManager":
        return _WorkspaceMembersManager(self)

    @staticmethod
    def from_context(*, client: Optional[_Client] = None) -> "_Workspace":
        """Look up the Workspace associated with the current context.

        This returns the Workspace that the active Modal credentials authenticate against
        (i.e., your active profile or the `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment
        variables). If called inside a Modal container, it returns the Workspace that the
        container is running in.
        """

        async def _load(
            self: "_Workspace", resolver: Resolver, load_context: LoadContext, existing_object_id: Optional[str]
        ):
            response = await load_context.client.stub.WorkspaceNameLookup(Empty())
            self._name = response.username or None
            self._client = load_context.client
            self._is_hydrated = True

        return _Workspace._from_loader(
            _load,
            "Workspace.from_context()",
            hydrate_lazily=True,
            load_context_overrides=LoadContext(client=client),
        )
