# Copyright Modal Labs 2023
import asyncio
import builtins
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Optional

from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import Message
from google.protobuf.wrappers_pb2 import StringValue
from synchronicity import classproperty

from modal_proto import api_pb2

from ._load_context import LoadContext
from ._object import _Object
from ._resolver import Resolver
from ._utils.name_utils import check_environment_name
from .client import _Client
from .config import config, logger
from .exception import InvalidError, WorkspaceManagementError


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
        client: Optional[_Client] = None,  # Optional client with Modal credentials
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
        client: Optional[_Client] = None,  # Optional client with Modal credentials
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
                is_another_app=True,
                rep=f"Environment.from_name({item.name!r})",
            )
            environments.append(env)
        return environments

    async def delete(
        self,
        name: str,  # Name of the Environment to delete
        *,
        client: Optional[_Client] = None,  # Optional client with Modal credentials
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


MemberRole = Literal["viewer", "contributor"]


def _role_to_proto(role: str) -> api_pb2.EnvironmentRole.ValueType:
    match role:
        case "viewer":
            return api_pb2.ENVIRONMENT_ROLE_VIEWER
        case "contributor":
            return api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR
        case _:
            raise InvalidError(f"Invalid Environment role: {role!r} (expected 'viewer' or 'contributor')")


def _role_from_proto(proto_value: int) -> MemberRole:
    match proto_value:
        case int(v) if v == api_pb2.ENVIRONMENT_ROLE_VIEWER:
            return "viewer"
        case int(v) if v == api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR:
            return "contributor"
        case _:
            raise ValueError(f"Unknown environment role: {proto_value}")


class _EnvironmentMembersManager:
    """mdmd:namespace
    Namespace with methods for managing the membership of a restricted Environment.

    See https://modal.com/docs/guide/rbac for more information on restricted Environments.
    """

    def __init__(self, environment: "_Environment"):
        """mdmd:hidden"""
        self._environment = environment

    async def list(self) -> dict[Literal["users", "service_users"], dict[str, MemberRole]]:
        """Return the members of a restricted Environment with their roles.

        **Examples:**

        ```python notest
        members = modal.Environment.from_name("my-restricted-env").members.list()
        print(members)
        # {
        #     "users": {"alice": "contributor", "bob": "viewer"},
        #     "service_users": {"alice-bot": "contributor"},
        # }
        ```
        """
        await self._environment.hydrate()
        req = api_pb2.EnvironmentGetManagedRequest(environment_id=self._environment.object_id)
        resp = await self._environment.client.stub.EnvironmentGetManaged(req)

        users: dict[str, MemberRole] = {}
        service_users: dict[str, MemberRole] = {}
        for principal in resp.principal_roles:
            role = _role_from_proto(principal.role)
            if principal.user_id:
                users[principal.user_name] = role
            elif principal.service_user_id:
                service_users[principal.service_user_name] = role

        return {"users": users, "service_users": service_users}

    async def update(
        self,
        *,
        users: Optional[Mapping[str, MemberRole]] = None,
        service_users: Optional[Mapping[str, MemberRole]] = None,
    ) -> None:
        """Add or modify roles for members of a restricted Environment.

        Each user or service user will be added to the Environment if not currently a member;
        if already a member, the user or service user's role will be updated.

        **Examples:**

        ```python notest
        env = modal.Environment.from_name("my-restricted-env")
        env.members.update(
            users={"alice": "contributor", "bob": "viewer"},
            service_users={"alice-bot": "contributor"},
        )
        ```
        """
        await self._environment.hydrate()
        users = users or {}
        service_users = service_users or {}

        req = api_pb2.EnvironmentGetManagedRequest(environment_id=self._environment.object_id)
        resp = await self._environment.client.stub.EnvironmentGetManaged(req)

        # Both current members and additional eligible workspace principals can be assigned a role
        user_name_to_id: dict[str, str] = {}
        service_user_name_to_id: dict[str, str] = {}
        for principal in [*resp.principal_roles, *resp.additional_roles]:
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
                role=_role_to_proto(role),
            )
        for name, role in service_users.items():
            if name not in service_user_name_to_id:
                raise InvalidError(f"Service user {name!r} not found in workspace")
            requests[f"Service user {name!r}"] = api_pb2.EnvironmentRoleSetRequest(
                environment_id=self._environment.object_id,
                service_user_id=service_user_name_to_id[name],
                role=_role_to_proto(role),
            )

        await self._dispatch_role_updates(requests)

    async def remove(
        self,
        *,
        users: Optional[Iterable[str]] = None,
        service_users: Optional[Iterable[str]] = None,
    ) -> None:
        """Remove members from a restricted Environment.

        **Examples:**

        ```python notest
        env = modal.Environment.from_name("my-restricted-env")
        env.members.remove(
            users=["alice"],
            service_users=["alice-bot"],
        )
        ```
        """
        await self._environment.hydrate()
        users = users or []
        service_users = service_users or []

        req = api_pb2.EnvironmentGetManagedRequest(environment_id=self._environment.object_id)
        resp = await self._environment.client.stub.EnvironmentGetManaged(req)

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
                raise InvalidError(f"User {name!r} is not a member of this Environment")
            requests[f"User {name!r}"] = api_pb2.EnvironmentRoleSetRequest(
                environment_id=self._environment.object_id,
                user_id=user_name_to_id[name],
                role=api_pb2.ENVIRONMENT_ROLE_UNSPECIFIED,
            )
        for name in service_users:
            if name not in service_user_name_to_id:
                raise InvalidError(f"Service user {name!r} is not a member of this Environment")
            requests[f"Service user {name!r}"] = api_pb2.EnvironmentRoleSetRequest(
                environment_id=self._environment.object_id,
                service_user_id=service_user_name_to_id[name],
                role=api_pb2.ENVIRONMENT_ROLE_UNSPECIFIED,
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
            header = f"{n} error{'s' if n != 1 else ''} occurred while updating Environment members:"
            details = "\n".join(f"  - {label}: {e}" for label, e in errors)
            raise WorkspaceManagementError(f"{header}\n{details}")


class _Environment(_Object, type_prefix="en"):
    _name: Optional[str] = None
    _settings: EnvironmentSettings

    def __init__(self):
        """mdmd:hidden"""
        raise RuntimeError(
            "`Environment(...)` constructor is not allowed. "
            "Use `Environment.from_name` or `Environment.from_context` instead."
        )

    @property
    def name(self) -> Optional[str]:
        return self._name

    @classproperty
    @classmethod
    def objects(cls) -> _EnvironmentManager:
        return _EnvironmentManager()

    @property
    def members(self) -> "_EnvironmentMembersManager":
        return _EnvironmentMembersManager(self)

    # TODO(michael) Keeping this private for now until we decide what else should be in it
    # And what the rules should be about updates / mutability
    # @property
    # def settings(self) -> EnvironmentSettings:
    #     return self._settings

    def _hydrate_metadata(self, metadata: Message):
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
        name: str, repr: str, create_if_missing: bool = False, client: Optional[_Client] = None
    ) -> "_Environment":
        async def _load(
            self: _Environment, resolver: Resolver, load_context: LoadContext, existing_object_id: Optional[str]
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
            is_another_app=True,
            hydrate_lazily=True,
            name=name,
            load_context_overrides=LoadContext(client=client),
        )

    @staticmethod
    def from_context(*, client: Optional[_Client] = None) -> "_Environment":
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
        client: Optional[_Client] = None,
    ) -> "_Environment":
        """Look up an Environment object using its name."""
        check_environment_name(name)
        return _Environment._get_or_create(
            name=name,
            repr=f"Environment.from_name({name!r})",
            create_if_missing=create_if_missing,
            client=client,
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


async def _delete_environment(name: str, client: Optional[_Client] = None):
    if client is None:
        client = await _Client.from_env()
    await client.stub.EnvironmentDelete(api_pb2.EnvironmentDeleteRequest(name=name))


async def _update_environment(
    current_name: str,
    *,
    new_name: Optional[str] = None,
    new_web_suffix: Optional[str] = None,
    client: Optional[_Client] = None,
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


async def _create_environment(name: str, client: Optional[_Client] = None):
    if client is None:
        client = await _Client.from_env()
    await client.stub.EnvironmentCreate(api_pb2.EnvironmentCreateRequest(name=name))


async def _list_environments(client: Optional[_Client] = None) -> list[api_pb2.EnvironmentListItem]:
    if client is None:
        client = await _Client.from_env()
    resp = await client.stub.EnvironmentList(Empty())
    return list(resp.items)


def ensure_env(environment_name: Optional[str] = None) -> str:
    """Override config environment with environment from environment_name

    This is necessary since a cli command that runs Modal code, without explicit
    environment specification wouldn't pick up the environment specified in a
    command line flag otherwise, e.g. when doing `modal run --env=foo`
    """
    if environment_name is not None:
        config.override_locally("environment", environment_name)

    return config.get("environment")
