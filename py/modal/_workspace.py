# Copyright Modal Labs 2025
import builtins
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from google.protobuf.empty_pb2 import Empty

from modal.exception import InvalidError
from modal_proto import api_pb2

from ._billing import BILLING_DOCSTRING, BillingReportItem
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

    @property
    def billing(self) -> "_WorkspaceBillingManager":
        return _WorkspaceBillingManager(self)

    @property
    def proxy_tokens(self) -> "_WorkspaceProxyTokenManager":
        return _WorkspaceProxyTokenManager(self)


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


class _WorkspaceProxyTokenManager:
    """mdmd:namespace
    Namespace with methods for managing the proxy tokens in a Workspace.

    See [the guide](https://modal.com/docs/guide/webhook-proxy-auth) for more information on proxy tokens.
    """

    def __init__(self, workspace: "_Workspace"):
        """mdmd:hidden"""
        self._workspace = workspace

    async def create(self) -> TokenData:
        """Create a new proxy token for the Workspace.

        Examples:
            ```python notest
            token = modal.Workspace.from_context().proxy_tokens.create()
            print(token.token_id, token.token_secret)
            ```
        """
        await self._workspace.hydrate()
        resp = await self._workspace.client.stub.WebhookTokenCreate(api_pb2.WebhookTokenCreateRequest())
        return TokenData(token_id=resp.token_id, token_secret=resp.token_secret)

    async def list(self, environment_name: Optional[str] = None) -> builtins.list[ProxyTokenInfo]:
        """List proxy tokens in the Workspace.

        Args:
            environment_name: When provided, list only the tokens associated with this environment.

        Examples:
            ```python notest
            ws = modal.Workspace.from_context()

            # List all proxy tokens in the Workspace
            tokens = ws.proxy_tokens.list()
            print([t.token_id for t in tokens])

            # List only the proxy tokens associated with a specific Environment
            env_tokens = ws.proxy_tokens.list(environment_name="prod")
            ```
        """
        await self._workspace.hydrate()
        if environment_name is None:
            resp = await self._workspace.client.stub.WebhookTokenList(Empty())
        else:
            resp = await self._workspace.client.stub.WebhookTokenListForEnvironment(
                api_pb2.WebhookTokenListForEnvironmentRequest(environment_name=environment_name)
            )
        return [
            ProxyTokenInfo(
                token_id=token.token_id,
                created_at=timestamp_to_localized_dt(token.created_at),
                scoped=token.scoped,
            )
            for token in resp.tokens
        ]

    async def allow(self, proxy_token_id: str, environment_name: str) -> None:
        """Allow a proxy token to authenticate requests to a given Environment.

        Args:
            proxy_token_id: The token ID (`wk-...`) to operate on.
            environment_name: The name of the environment to allow access to.

        Examples:
            ```python notest
            ws = modal.Workspace.from_context()
            token = ws.proxy_tokens.create()
            ws.proxy_tokens.allow(token.token_id, "prod")
            ```
        """
        await self._workspace.hydrate()
        environment_id = await self._environment_id(environment_name)
        req = api_pb2.WebhookTokenEnvironmentAddRequest(token_id=proxy_token_id, environment_id=environment_id)
        await self._workspace.client.stub.WebhookTokenEnvironmentAdd(req)

    async def revoke(self, proxy_token_id: str, environment_name: str) -> None:
        """Revoke a proxy token's access to a given Environment.

        The proxy token is not deleted, and it will continue to authenticate requests to any
        other Environments it is associated with.

        Args:
            proxy_token_id: The token ID (`wk-...`) to operate on.
            environment_name: The name of the environment to revoke access from.

        Examples:
            ```python notest
            ws = modal.Workspace.from_context()
            ws.proxy_tokens.revoke(token_id, "prod")
            ```
        """
        await self._workspace.hydrate()
        environment_id = await self._environment_id(environment_name)
        req = api_pb2.WebhookTokenEnvironmentRemoveRequest(token_id=proxy_token_id, environment_id=environment_id)
        await self._workspace.client.stub.WebhookTokenEnvironmentRemove(req)

    async def delete(self, proxy_token_id: str) -> None:
        """Delete a proxy token from the Workspace.

        This cannot be reverted. Any clients currently using the token will immediately
        lose access to associated resources.

        Args:
            proxy_token_id: The token ID (`wk-...`) to delete.

        Examples:
            ```python notest
            modal.Workspace.from_context().proxy_tokens.delete(token_id)
            ```
        """
        await self._workspace.hydrate()
        await self._workspace.client.stub.WebhookTokenDelete(api_pb2.TokenDeleteRequest(token_id=proxy_token_id))

    async def _environment_id(self, environment_name: str) -> str:
        # The environment-association RPCs key on environment ID, so resolve the name first.
        from ._environments import _Environment

        environment = await _Environment.from_name(environment_name, client=self._workspace.client).hydrate()
        return environment.object_id


class _WorkspaceBillingManager:
    """mdmd:namespace
    Namespace for Workspace billing APIs.
    """

    def __init__(self, workspace: _Workspace):
        """mdmd:hidden"""
        self._workspace = workspace

    async def report(
        self,
        *,
        start: datetime,  # Start of the report, inclusive
        end: datetime | None = None,  # End of the report, exclusive
        resolution: str = "d",  # Resolution, e.g. "d" for daily or "h" for hourly
        tag_names: list[str] | None = None,  # Optional additional metadata to include
    ) -> list[BillingReportItem]:
        (
            """Return a report of workspace usage by object and time.

            The result will be a list of dataclasses for each interval (determined by `resolution`)
            between the `start` and `end` limits. Each item represents a single (Modal object, time interval)
            pair that billing can be attributed to (e.g., an App) along with metadata (including user-defined
            tags) to identify that object. The dataclass also contains a breakdown of the cost value
            attributed to individual resources (for an App, this can be CPU, Memory, specific GPU types,
            etc.). The specific resource types included in the breakdown are subject to change as
            Modal's billing model evolves.

            It's also possible to generate reports using the
            [`modal billing report`](https://modal.com/docs/cli/latest/billing#modal-billing-report)
            CLI command. The CLI has a few convenience features for generating reports across relative
            time ranges.

            """
            + BILLING_DOCSTRING
        )

        if tag_names is None:
            tag_names = []

        if end is None:
            end = datetime.now(timezone.utc)

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        elif start.tzinfo != timezone.utc:
            raise InvalidError("Timezone-aware 'start' parameter must be in UTC.")

        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        elif end.tzinfo != timezone.utc:
            raise InvalidError("Timezone-aware 'end' parameter must be in UTC.")

        if not self._workspace.is_hydrated:
            await self._workspace.hydrate()

        request = api_pb2.WorkspaceBillingReportRequest(
            resolution=resolution,
            tag_names=tag_names,
        )
        request.start_timestamp.FromDatetime(start)
        request.end_timestamp.FromDatetime(end)

        return [
            BillingReportItem._from_proto(pb_item)
            async for pb_item in self._workspace.client.stub.WorkspaceBillingReport.unary_stream(request)
        ]
