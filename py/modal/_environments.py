# Copyright Modal Labs 2023
import asyncio
import builtins
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import Message
from google.protobuf.wrappers_pb2 import StringValue
from synchronicity import classproperty

from modal_proto import api_pb2

from ._load_context import LoadContext
from ._object import _Object
from ._resolver import Resolver
from ._utils.deprecation import deprecation_warning
from ._utils.name_utils import check_environment_name
from ._utils.time_utils import is_utc_month_aligned, parse_billing_cycle
from .client import _Client
from .config import config, logger
from .exception import InvalidError, WorkspaceManagementError
from .types import BillingReportItem, EnvironmentBillingSummary


@dataclass(frozen=True)
class EnvironmentSettings:
    image_builder_version: str
    webhook_suffix: str


class _EnvironmentManager:
    """Namespace with methods for managing Environment objects."""

    async def create(
        self,
        name: str,  # Name to use for the new Environment
        *,
        restricted: bool = False,  # If True, enable RBAC restrictions on the Environment
        client: _Client | None = None,  # Optional client with Modal credentials
    ) -> None:
        """Create a new Environment.

        **Examples:**

        ```python notest
        modal.Environment.objects.create("my-environment")
        ```
        """
        check_environment_name(name)
        client = await _Client.from_env() if client is None else client
        await client.stub.EnvironmentCreate(api_pb2.EnvironmentCreateRequest(name=name, is_managed=restricted))

    async def list(
        self,
        *,
        client: _Client | None = None,  # Optional client with Modal credentials
    ) -> builtins.list["_Environment"]:
        """Return a list of hydrated Environment objects.

        **Examples:**

        ```python notest
        environments = modal.Environment.objects.list()
        print([e.name for e in environments])
        ```
        """
        client = await _Client.from_env() if client is None else client
        resp = await client.stub.EnvironmentList(Empty())
        environments = []
        for item in resp.items:
            metadata = api_pb2.EnvironmentMetadata(
                name=item.name,
                settings=api_pb2.EnvironmentSettings(webhook_suffix=item.webhook_suffix),
            )
            env = _Environment._new_hydrated(
                item.environment_id,
                client,
                metadata,
                skip_reload=True,
                rep=f"Environment.from_name({item.name!r})",
            )
            environments.append(env)
        return environments

    async def delete(
        self,
        name: str,  # Name of the Environment to delete
        *,
        client: _Client | None = None,  # Optional client with Modal credentials
    ) -> None:
        """Delete a named Environment.

        Warning: This is irreversible and will transitively delete all objects in the Environment.

        **Examples:**

        ```python notest
        modal.Environment.objects.delete("my-environment")
        ```
        """
        client = await _Client.from_env() if client is None else client
        await client.stub.EnvironmentDelete(api_pb2.EnvironmentDeleteRequest(name=name))


class _EnvironmentRolesManager:
    """mdmd:namespace"""

    def __init__(self, environment: "_Environment"):
        """mdmd:hidden"""
        self._environment = environment

    async def list(self) -> dict[Literal["users", "service_users"], dict[str, str]]:
        """Enumerate the Environment Role for each user and service user in the workspace.

        **Examples:**

        ```python notest
        roles = modal.Environment.from_name("my-env").roles.list()
        print(roles)
        # {
        #     "users": {"alice": "contributor", "bob": "viewer", "carol": "contributor"},
        #     "service_users": {"alice-bot": "contributor", "ops-bot": "viewer", "ci-bot": "no-access"},
        # }
        ```
        """
        await self._environment.hydrate()
        req = api_pb2.EnvironmentGetRolesRequest(environment_id=self._environment.object_id)
        resp = await self._environment.client.stub.EnvironmentGetRoles(req)

        users: dict[str, str] = {}
        service_users: dict[str, str] = {}
        for principal in resp.principal_roles:
            if principal.user_id:
                users[principal.user_name] = principal.role_str
            elif principal.service_user_id:
                service_users[principal.service_user_name] = principal.role_str

        return {"users": users, "service_users": service_users}

    async def update(
        self,
        *,
        users: Mapping[str, str] | None = None,
        service_users: Mapping[str, str] | None = None,
    ) -> None:
        """Update the Environment Role of users and service users.

        Each role is one of 'contributor', 'viewer', or 'no-access'. Service users can be
        assigned a role on any Environment, while workspace members can only be assigned a
        role on restricted Environments.

        **Examples:**

        ```python notest
        env = modal.Environment.from_name("my-restricted-env")
        env.roles.update(
            users={"alice": "contributor", "bob": "viewer"},
            service_users={"alice-bot": "contributor"},
        )
        ```
        """
        await self._environment.hydrate()
        users = users or {}
        service_users = service_users or {}

        req = api_pb2.EnvironmentGetRolesRequest(environment_id=self._environment.object_id)
        resp = await self._environment.client.stub.EnvironmentGetRoles(req)

        # EnvironmentGetRoles returns every workspace principal
        user_name_to_id: dict[str, str] = {}
        service_user_name_to_id: dict[str, str] = {}
        for principal in resp.principal_roles:
            if principal.user_id:
                user_name_to_id[principal.user_name] = principal.user_id
            elif principal.service_user_id:
                service_user_name_to_id[principal.service_user_name] = principal.service_user_id

        requests: dict[str, api_pb2.EnvironmentRoleSetRequest] = {}
        for name, role in users.items():
            if name not in user_name_to_id:
                raise InvalidError(f"User {name!r} not found in workspace")
            requests[f"User {name!r}"] = api_pb2.EnvironmentRoleSetRequest(
                environment_id=self._environment.object_id,
                user_id=user_name_to_id[name],
                role_str=role,
            )
        for name, role in service_users.items():
            if name not in service_user_name_to_id:
                raise InvalidError(f"Service user {name!r} not found in workspace")
            requests[f"Service user {name!r}"] = api_pb2.EnvironmentRoleSetRequest(
                environment_id=self._environment.object_id,
                service_user_id=service_user_name_to_id[name],
                role_str=role,
            )

        await self._dispatch_role_updates(requests)

    async def _dispatch_role_updates(self, requests: dict[str, api_pb2.EnvironmentRoleSetRequest]) -> None:
        """Send batched EnvironmentRoleSet RPCs and report all errors encountered."""
        results = await asyncio.gather(
            *(self._environment.client.stub.EnvironmentRoleSet(req) for req in requests.values()),
            return_exceptions=True,
        )
        errors = [(label, result) for label, result in zip(requests.keys(), results) if isinstance(result, Exception)]
        if errors:
            n = len(errors)
            header = f"{n} error{'s' if n != 1 else ''} occurred while updating Environment roles:"
            details = "\n".join(f"  - {label}: {e}" for label, e in errors)
            raise WorkspaceManagementError(f"{header}\n{details}")


# Deprecated alias for `_EnvironmentRolesManager`; each method warns and delegates.
class _EnvironmentMembersManager:
    """mdmd:hidden"""

    def __init__(self, environment: "_Environment"):
        """mdmd:hidden"""
        self._environment = environment

    async def list(self) -> dict[Literal["users", "service_users"], dict[str, str]]:
        deprecation_warning(
            (2026, 7, 23),
            "`Environment.members.list()` is deprecated; use `Environment.roles.list()` instead.",
        )
        return await _EnvironmentRolesManager(self._environment).list()

    async def update(
        self,
        *,
        users: Mapping[str, str] | None = None,
        service_users: Mapping[str, str] | None = None,
    ) -> None:
        deprecation_warning(
            (2026, 7, 23),
            "`Environment.members.update()` is deprecated; use `Environment.roles.update()` instead.",
        )
        await _EnvironmentRolesManager(self._environment).update(users=users, service_users=service_users)

    async def remove(
        self,
        *,
        users: Iterable[str] | None = None,
        service_users: Iterable[str] | None = None,
    ) -> None:
        """Remove the Environment Role of users and service users, reverting them to the default."""
        deprecation_warning(
            (2026, 7, 23),
            "`Environment.members.remove()` is deprecated. Environment Roles are now explicit; "
            "set a role (e.g. 'no-access') with `Environment.roles.update()` instead.",
        )
        await self._environment.hydrate()
        users = users or []
        service_users = service_users or []

        req = api_pb2.EnvironmentGetRolesRequest(environment_id=self._environment.object_id)
        resp = await self._environment.client.stub.EnvironmentGetRoles(req)

        user_name_to_id: dict[str, str] = {}
        service_user_name_to_id: dict[str, str] = {}
        for principal in resp.principal_roles:
            if principal.user_id:
                user_name_to_id[principal.user_name] = principal.user_id
            elif principal.service_user_id:
                service_user_name_to_id[principal.service_user_name] = principal.service_user_id

        requests: dict[str, api_pb2.EnvironmentRoleSetRequest] = {}
        for name in users:
            if name not in user_name_to_id:
                raise InvalidError(f"User {name!r} not found in workspace")
            requests[f"User {name!r}"] = api_pb2.EnvironmentRoleSetRequest(
                environment_id=self._environment.object_id,
                user_id=user_name_to_id[name],
                role=api_pb2.ENVIRONMENT_ROLE_UNSPECIFIED,
            )
        for name in service_users:
            if name not in service_user_name_to_id:
                raise InvalidError(f"Service user {name!r} not found in workspace")
            requests[f"Service user {name!r}"] = api_pb2.EnvironmentRoleSetRequest(
                environment_id=self._environment.object_id,
                service_user_id=service_user_name_to_id[name],
                role=api_pb2.ENVIRONMENT_ROLE_UNSPECIFIED,
            )

        await _EnvironmentRolesManager(self._environment)._dispatch_role_updates(requests)


class _Environment(_Object, type_prefix="en"):
    _name: str | None = None
    _settings: EnvironmentSettings

    def __init__(self):
        """mdmd:hidden"""
        raise RuntimeError(
            "`Environment(...)` constructor is not allowed. "
            "Use `Environment.from_name` or `Environment.from_context` instead."
        )

    @property
    def name(self) -> str | None:
        return self._name

    @classproperty
    @classmethod
    def objects(cls) -> _EnvironmentManager:
        return _EnvironmentManager()

    @property
    def roles(self) -> "_EnvironmentRolesManager":
        """Namespace with methods for managing the Environment Roles of users and service users.

        See https://modal.com/docs/guide/rbac for more information on Environment Roles.
        """
        return _EnvironmentRolesManager(self)

    @property
    def members(self) -> "_EnvironmentMembersManager":
        """mdmd:hidden"""
        # Deprecated alias for `Environment.roles`.
        return _EnvironmentMembersManager(self)

    # TODO(michael) Keeping this private for now until we decide what else should be in it
    # And what the rules should be about updates / mutability
    # @property
    # def settings(self) -> EnvironmentSettings:
    #     return self._settings

    def _hydrate_metadata(self, metadata: Message | None):
        # Overridden concrete implementation of base class method
        assert metadata and isinstance(metadata, api_pb2.EnvironmentMetadata)
        self._name = metadata.name or None

        # Is there a simpler way to go Message -> Dataclass?
        self._settings = EnvironmentSettings(
            image_builder_version=metadata.settings.image_builder_version,
            webhook_suffix=metadata.settings.webhook_suffix,
        )

    @staticmethod
    def _get_or_create(
        name: str, repr: str, create_if_missing: bool = False, client: _Client | None = None
    ) -> "_Environment":
        async def _load(
            self: _Environment, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None
        ):
            request = api_pb2.EnvironmentGetOrCreateRequest(
                deployment_name=name,
                object_creation_type=(
                    api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING
                    if create_if_missing
                    else api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED
                ),
            )
            response = await load_context.client.stub.EnvironmentGetOrCreate(request)
            logger.debug(f"Created environment with id {response.environment_id}")
            self._hydrate(response.environment_id, load_context.client, response.metadata)

        return _Environment._from_loader(
            _load,
            repr,
            skip_reload=True,
            hydrate_lazily=True,
            name=name,
            load_context_overrides=LoadContext(client=client),
        )

    @staticmethod
    def from_context(*, client: _Client | None = None) -> "_Environment":
        """Look up an Environment object using the current context.

        This method returns the Environment that is defined by the local configuration
        (i.e., your active profile or the `MODAL_ENVIRONMENT` environment variable), or
        it fetches the default environment from the server when not defined locally.
        If called inside a Modal container, it will return the Environment that container
        is associated with.

        """
        name = config.get("environment") or ""  # null string falls back to server default
        return _Environment._get_or_create(
            name=name,
            repr="Environment.from_context()",
            create_if_missing=False,
            client=client,
        )

    @staticmethod
    def from_name(
        name: str,
        *,
        create_if_missing: bool = False,
        client: _Client | None = None,
    ) -> "_Environment":
        """Look up an Environment object using its name."""
        check_environment_name(name)
        return _Environment._get_or_create(
            name=name,
            repr=f"Environment.from_name({name!r})",
            create_if_missing=create_if_missing,
            client=client,
        )

    @property
    def billing(self) -> "_EnvironmentBillingManager":
        """Namespace for Environment billing APIs."""
        return _EnvironmentBillingManager(self)


class _EnvironmentBillingManager:
    """mdmd:namespace"""

    def __init__(self, environment: _Environment):
        """mdmd:ignore"""
        self._environment = environment

    async def report(
        self,
        *,
        start: datetime,  # Start of the report, inclusive
        end: datetime | None = None,  # End of the report, exclusive
        resolution: str = "d",  # Resolution, e.g. "d" for daily or "h" for hourly
        tag_names: list[str] | None = None,  # Optional additional metadata to include
    ) -> list[BillingReportItem]:
        """Return a cost report for Environment usage, broken down by object and time.

        Args:
            start: Start of the report, inclusive and rounded to the beginning of the interval.
                Must be in UTC or timezone-naive (interpreted as UTC).
            end: End of the report, exclusive. Must be in UTC or timezone-naive. Partial final
                intervals will be excluded from the report.
            resolution: Resolution, e.g. "d" for daily or "h" for hourly.
            tag_names: List of tag names; each row will include the tag name and value in use
                for that object during the relevant time interval. Pass `["*"]` to include all
                tags in the report.

        Returns:
            A list of `BillingReportItem` dataclasses. Each item reports the cost attributed to
            a specific Modal object during a given time interval. Cost is further broken down by
            the resource type that generated it (e.g. CPU, Memory, specific GPU usage).
            Note that the specific resource types included in the breakdown are subject to change
            as Modal's billing model evolves.

        See also:
            - [`modal environment billing report`](https://modal.com/docs/cli/latest/environment#modal-environment-billing-report):
              An environment report CLI that has convenience features around relative time range queries
              and JSON/CSV output.
            - [`Workspace.billing.report()`](https://modal.com/docs/sdk/py/latest/Workspace#billingreport):
              An analogous report API for the entire Workspace.

        """
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

        if not self._environment.is_hydrated:
            await self._environment.hydrate()

        request = api_pb2.WorkspaceBillingReportRequest(
            resolution=resolution,
            tag_names=tag_names,
            environment_ids=[self._environment.object_id],
        )
        request.start_timestamp.FromDatetime(start)
        request.end_timestamp.FromDatetime(end)

        return [
            BillingReportItem._from_proto(pb_item)
            async for pb_item in self._environment.client.stub.WorkspaceBillingReport.unary_stream(request)
        ]

    async def summary(
        self,
        cycle: str | datetime | None = None,  # Start of the summary, inclusive
    ) -> EnvironmentBillingSummary:
        """Return a summary of environment cost over a single billing cycle determined by `cycle`.

        Unlike the analogous `Workspace.billing.summary()`, this API only emits metered cost
        information. This is because billing adjustments due to credits, free storage, etc. are
        applied at the Workspace level, and thus cannot be attributed to individual Environments.

        Args:
            cycle: Start of the summary, inclusive. Must be the first of a month, and must be in UTC
                or timezone-naive (interpreted as UTC). If provided as a string, it must either be
                formatted as an ISO 8601 month (YYYY-MM), or must be one of the convenience spellings
                "this month" or "last month". If not provided, `cycle` defaults to the first of the
                current month (in which case a summary is generated for the current billing cycle).

        Returns:
            A single `EnvironmentBillingSummary` dataclass containing the following fields:
            - `metered_cost` representing cost before any adjustments, and
            - `metered_cost_breakdown` containing a breakdown of that cost by the Modal resources
              that generated it. The exact keys of this are subject to change as Modal's billing
              model evolves.

            All values are reported as `decimal.Decimal`s.

        See also:
            - [`modal environment billing summary`](https://modal.com/docs/cli/latest/billing#modal-environment-billing-summary):
              An environment summary CLI that has convenience features around relative time range queries.
            - [`Environment.billing.report()`](https://modal.com/docs/sdk/py/latest/Environment#billingreport):
              An analogous report API that is scoped to a specific Environment.
        """

        if cycle is None:
            cycle = datetime.now(timezone.utc).replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        elif isinstance(cycle, str):
            cycle = parse_billing_cycle(cycle)

        if cycle.tzinfo is None:
            cycle = cycle.replace(tzinfo=timezone.utc)
        elif cycle.tzinfo != timezone.utc:
            raise InvalidError("Timezone-aware 'cycle' parameter must be in UTC.")

        if not is_utc_month_aligned(cycle):
            raise InvalidError("Provided 'cycle' parameter must be the first of a month.")

        if cycle > datetime.now(timezone.utc):
            raise InvalidError("Provided 'cycle' parameter cannot be in the future.")

        if not self._environment.is_hydrated:
            await self._environment.hydrate()

        request = api_pb2.EnvironmentBillingSummaryRequest(environment_id=self._environment.object_id)
        request.start_timestamp.FromDatetime(cycle)

        return EnvironmentBillingSummary._from_proto(
            await self._environment.client.stub.EnvironmentBillingSummary(request)
        )


ENVIRONMENT_CACHE: dict[str, _Environment] = {}


async def _get_environment_cached(name: str, client: _Client) -> _Environment:
    if name in ENVIRONMENT_CACHE:
        return ENVIRONMENT_CACHE[name]
    if name:
        environment = await _Environment.from_name(name, client=client).hydrate()
    else:
        environment = await _Environment.from_context(client=client).hydrate()
    ENVIRONMENT_CACHE[name] = environment
    return environment


# The following internal functions are functionally public API as users have come to
# depend on them while we did not have a proper Environment API. We can deprecate them
# and migrate users to the new object-oriented API, but that should happen gracefully.


async def _delete_environment(name: str, client: _Client | None = None):
    if client is None:
        client = await _Client.from_env()
    await client.stub.EnvironmentDelete(api_pb2.EnvironmentDeleteRequest(name=name))


async def _update_environment(
    current_name: str,
    *,
    new_name: str | None = None,
    new_web_suffix: str | None = None,
    client: _Client | None = None,
):
    new_name_pb2 = None
    new_web_suffix_pb2 = None
    if new_name is not None:
        if len(new_name) < 1:
            raise ValueError("The new environment name cannot be empty")

        new_name_pb2 = StringValue(value=new_name)

    if new_web_suffix is not None:
        new_web_suffix_pb2 = StringValue(value=new_web_suffix)

    update_payload = api_pb2.EnvironmentUpdateRequest(
        current_name=current_name, name=new_name_pb2, web_suffix=new_web_suffix_pb2
    )
    if client is None:
        client = await _Client.from_env()
    await client.stub.EnvironmentUpdate(update_payload)


async def _create_environment(name: str, client: _Client | None = None):
    if client is None:
        client = await _Client.from_env()
    await client.stub.EnvironmentCreate(api_pb2.EnvironmentCreateRequest(name=name))


async def _list_environments(client: _Client | None = None) -> list[api_pb2.EnvironmentListItem]:
    if client is None:
        client = await _Client.from_env()
    resp = await client.stub.EnvironmentList(Empty())
    return list(resp.items)


def ensure_env(environment_name: str | None = None) -> str:
    """Override config environment with environment from environment_name

    This is necessary since a cli command that runs Modal code, without explicit
    environment specification wouldn't pick up the environment specified in a
    command line flag otherwise, e.g. when doing `modal run --env=foo`
    """
    if environment_name is not None:
        config.override_locally("environment", environment_name)

    return config.get("environment") or ""
