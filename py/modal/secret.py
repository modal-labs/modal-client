# Copyright Modal Labs 2022
import builtins
import os
from datetime import datetime
from typing import Callable

from google.protobuf.message import Message
from synchronicity import classproperty

from modal_proto import api_pb2

from ._load_context import LoadContext
from ._object import _get_environment_name, _Object, live_method
from ._resolver import Resolver
from ._runtime.execution_context import is_local
from ._utils.async_utils import synchronize_api
from ._utils.name_utils import check_object_name
from ._utils.time_utils import as_timestamp, timestamp_to_localized_dt
from .client import _Client
from .exception import AlreadyExistsError, InvalidError, NotFoundError
from .types import SecretInfo

ENV_DICT_WRONG_TYPE_ERR = "the env_dict argument to Secret has to be a dict[str, Union[str, None]]"


class _SecretManager:
    """Namespace with methods for managing named Secret objects."""

    async def create(
        self,
        name: str,
        env_dict: dict[str, str],
        *,
        allow_existing: bool = False,
        environment_name: str | None = None,
        client: _Client | None = None,
    ) -> None:
        """Create a new named Secret in the workspace environment.

        This does not return a local handle; use `modal.Secret.from_name` to look up the Secret after creation.

        Added in v1.1.2.

        Args:
            name: Name for the new Secret.
            env_dict: Environment variable keys and values stored in the Secret.
            allow_existing: If True, do nothing when a Secret with this name already exists (existing values are kept).
            environment_name: Environment to create in; defaults to the active environment.
            client: Modal client to use; defaults to `Client.from_env()` when omitted.

        Examples:
            ```python notest
            contents = {"MY_KEY": "my-value", "MY_OTHER_KEY": "my-other-value"}
            modal.Secret.objects.create("my-secret", contents)
            ```

            Secrets will be created in the active environment, or another one can be specified:

            ```python notest
            modal.Secret.objects.create("my-secret", contents, environment_name="dev")
            ```

            By default, an error will be raised if the Secret already exists, but passing
            `allow_existing=True` will make the creation attempt a no-op in this case.
            If the `env_dict` data differs from the existing Secret, it will be ignored.

            ```python notest
            modal.Secret.objects.create("my-secret", contents, allow_existing=True)
            ```

            Note that this method does not return a local instance of the Secret. You can use
            `modal.Secret.from_name` to perform a lookup after creation.
        """
        check_object_name(name, "Secret")
        client = await _Client.from_env() if client is None else client
        object_creation_type = (
            api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING
            if allow_existing
            else api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS
        )
        req = api_pb2.SecretGetOrCreateRequest(
            deployment_name=name,
            environment_name=_get_environment_name(environment_name),
            object_creation_type=object_creation_type,
            env_dict=env_dict,
        )
        try:
            await client.stub.SecretGetOrCreate(req)
        except AlreadyExistsError:
            if not allow_existing:
                raise

    async def list(
        self,
        *,
        max_objects: int | None = None,
        created_before: datetime | str | None = None,
        environment_name: str = "",
        client: _Client | None = None,
    ) -> builtins.list["_Secret"]:
        """List named Secrets in the workspace environment as hydrated handles.

        Results are ordered newest to oldest. By default, all matching Secrets are returned.

        Added in v1.1.2.

        Args:
            max_objects: Maximum number of Secrets to return.
            created_before: Only include Secrets created before this time (datetime or ISO date string).
            environment_name: Environment to list from; defaults to the active environment.
            client: Modal client to use; defaults to `Client.from_env()` when omitted.

        Returns:
            Hydrated `Secret` objects for each named Secret in the listing.

        Examples:
            ```python
            secrets = modal.Secret.objects.list()
            print([s.name for s in secrets])
            ```

            Secrets will be retrieved from the active environment, or another one can be specified:

            ```python notest
            dev_secrets = modal.Secret.objects.list(environment_name="dev")
            ```

            By default, all named Secrets are returned, newest to oldest. It's also possible to limit the
            number of results and to filter by creation date:

            ```python
            secrets = modal.Secret.objects.list(max_objects=10, created_before="2025-01-01")
            ```
        """
        client = await _Client.from_env() if client is None else client
        if max_objects is not None and max_objects < 0:
            raise InvalidError("max_objects cannot be negative")

        items: list[api_pb2.SecretListItem] = []

        async def retrieve_page(created_before: float) -> bool:
            max_page_size = 100 if max_objects is None else min(100, max_objects - len(items))
            pagination = api_pb2.ListPagination(max_objects=max_page_size, created_before=created_before)
            req = api_pb2.SecretListRequest(
                environment_name=_get_environment_name(environment_name), pagination=pagination
            )
            resp = await client.stub.SecretList(req)
            items.extend(resp.items)
            finished = (len(resp.items) < max_page_size) or (max_objects is not None and len(items) >= max_objects)
            return finished

        finished = await retrieve_page(as_timestamp(created_before))
        while True:
            if finished:
                break
            finished = await retrieve_page(items[-1].metadata.creation_info.created_at)

        secrets = [
            _Secret._new_hydrated(
                item.secret_id,
                client,
                item.metadata,
                skip_reload=True,
                rep=_Secret._repr(item.label, environment_name),
            )
            for item in items
        ]
        return secrets[:max_objects] if max_objects is not None else secrets

    async def delete(
        self,
        name: str,
        *,
        allow_missing: bool = False,
        environment_name: str | None = None,
        client: _Client | None = None,
    ):
        """Delete a named Secret entirely.

        Deletion is irreversible and affects any Apps using this Secret.

        Added in v1.1.2.

        Args:
            name: Name of the Secret to delete.
            allow_missing: If True, do nothing when the Secret does not exist.
            environment_name: Environment to delete from; defaults to the active environment.
            client: Modal client to use; defaults to `Client.from_env()` when omitted.

        Examples:
            ```python notest
            await modal.Secret.objects.delete("my-secret")
            ```

            Secrets will be deleted from the active environment, or another one can be specified:

            ```python notest
            await modal.Secret.objects.delete("my-secret", environment_name="dev")
            ```
        """
        try:
            obj = await _Secret.from_name(name, environment_name=environment_name).hydrate(client)
        except NotFoundError:
            if not allow_missing:
                raise
        else:
            req = api_pb2.SecretDeleteRequest(secret_id=obj.object_id)
            await obj._client.stub.SecretDelete(req)


SecretManager = synchronize_api(_SecretManager)


async def _load_from_env_dict(instance: "_Secret", load_context: LoadContext, env_dict: dict[str, str]):
    """helper method for loaders .from_dict and .from_dotenv etc."""
    if load_context.app_id is not None:
        req = api_pb2.SecretGetOrCreateRequest(
            object_creation_type=api_pb2.OBJECT_CREATION_TYPE_ANONYMOUS_OWNED_BY_APP,
            env_dict=env_dict,
            app_id=load_context.app_id,
            environment_name=load_context.environment_name,
        )
    else:
        req = api_pb2.SecretGetOrCreateRequest(
            object_creation_type=api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL,
            env_dict=env_dict,
            environment_name=load_context.environment_name,
        )

    resp = await load_context.client.stub.SecretGetOrCreate(req)
    instance._hydrate(resp.secret_id, load_context.client, resp.metadata)


class _Secret(_Object, type_prefix="st"):
    """Secrets provide a dictionary of environment variables for images.

    Secrets are a secure way to add credentials and other sensitive information
    to the containers your functions run in. You can create and edit secrets on
    [the dashboard](https://modal.com/secrets), or programmatically from Python code.

    See [the secrets guide page](https://modal.com/docs/guide/secrets) for more information.
    """

    _metadata: api_pb2.SecretMetadata | None = None
    _load_env_dict: Callable[[], dict[str, str]] | None = None

    @classproperty
    @classmethod
    def objects(cls) -> _SecretManager:
        return _SecretManager()

    @property
    def name(self) -> str | None:
        return self._name

    def _hydrate_metadata(self, metadata: Message | None):
        if metadata:
            assert isinstance(metadata, api_pb2.SecretMetadata)
            self._metadata = metadata
            self._name = metadata.name

    def _get_metadata(self) -> api_pb2.SecretMetadata:
        assert self._metadata
        return self._metadata

    @staticmethod
    def from_dict(
        env_dict: dict[str, str | None] = {},
    ) -> "_Secret":
        """Create a Secret from a dictionary of environment variable names to string values.

        Values may be ``None``; those keys are omitted from the Secret.

        Args:
            env_dict: Mapping of variable names to values (or ``None`` to skip a key).

        Returns:
            A lazy `Secret` handle backed by the given key-value pairs.

        Examples:
            ```python
            @app.function(secrets=[modal.Secret.from_dict({"FOO": "bar"})])
            def run():
                print(os.environ["FOO"])
            ```
        """
        if not isinstance(env_dict, dict):
            raise InvalidError(ENV_DICT_WRONG_TYPE_ERR)

        env_dict_filtered: dict[str, str] = {k: v for k, v in env_dict.items() if v is not None}
        if not all(isinstance(k, str) for k in env_dict_filtered.keys()):
            raise InvalidError(ENV_DICT_WRONG_TYPE_ERR)
        if not all(isinstance(v, str) for v in env_dict_filtered.values()):
            raise InvalidError(ENV_DICT_WRONG_TYPE_ERR)

        def _load_env_dict() -> dict[str, str]:
            return env_dict_filtered

        rep = f"Secret.from_dict([{', '.join(env_dict.keys())}])"
        # TODO: scoping - these should probably not be lazily hydrated without having an app and/or sandbox association
        return _Secret._from_load_env_dict(
            _load_env_dict, rep, hydrate_lazily=True, load_context_overrides=LoadContext.empty()
        )

    @staticmethod
    def from_local_environ(
        env_keys: list[str],
    ) -> "_Secret":
        """Build a Secret from the current process environment (local runs only).

        In remote execution, returns an empty Secret.

        Args:
            env_keys: Names of environment variables to copy into the Secret.

        Returns:
            A `Secret` containing the resolved variables (or empty when not local).
        """

        if is_local():
            try:
                return _Secret.from_dict({k: os.environ[k] for k in env_keys})
            except KeyError as exc:
                missing_key = exc.args[0]
                raise InvalidError(
                    f"Could not find local environment variable '{missing_key}' for Secret.from_local_environ"
                )

        return _Secret.from_dict({})

    @staticmethod
    def from_dotenv(path=None, *, filename=".env", client: _Client | None = None) -> "_Secret":
        """Load environment variables from a `.env` file into a Secret.

        With no `path`, searches from the current working directory (not the caller's file path).
        With `path` set, walks upward from that file or directory to find `filename`.

        Args:
            path: File or directory to search from; omit to search from the process cwd.
            filename: Name of the env file to find (default ``.env``).
            client: Modal client used when hydrating the Secret.

        Examples:
            ```python
            @app.function(secrets=[modal.Secret.from_dotenv(__file__)])
            def run():
                print(os.environ["USERNAME"])  # Assumes USERNAME is defined in your .env file
            ```

            ```python
            @app.function(secrets=[modal.Secret.from_dotenv(filename=".env-dev")])
            def run():
                ...
            ```

        Returns:
            A lazy `Secret` handle whose values are loaded from the resolved `.env` file.
        """

        def _load_env_dict() -> dict[str, str]:
            try:
                from dotenv import dotenv_values, find_dotenv
                from dotenv.main import _walk_to_root
            except ImportError:
                raise ImportError(
                    "Need the `dotenv` package installed. You can install it by running `pip install python-dotenv`."
                )

            if path is not None:
                # This basically implements the logic in find_dotenv
                for dirname in _walk_to_root(path):
                    check_path = os.path.join(dirname, filename)
                    if os.path.isfile(check_path):
                        dotenv_path = check_path
                        break
                else:
                    dotenv_path = ""
            else:
                # TODO(erikbern): dotenv tries to locate .env files based on location of the file in the stack frame.
                # Since the modal code "intermediates" this, a .env file in user's local directory won't be picked up.
                # To simplify this, we just support the cwd and don't do any automatic path inference.
                dotenv_path = find_dotenv(filename, usecwd=True)

            return {k: v or "" for k, v in dotenv_values(dotenv_path).items()}

        return _Secret._from_load_env_dict(
            _load_env_dict,
            "Secret.from_dotenv()",
            hydrate_lazily=True,
            load_context_overrides=LoadContext(client=client),
        )

    @classmethod
    def _from_load_env_dict(cls, load_env_dict: Callable[[], dict[str, str]], *args, **kwargs) -> "_Secret":
        async def _load(self: _Secret, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None):
            await _load_from_env_dict(self, load_context, load_env_dict())

        instance = _Secret._from_loader(_load, *args, **kwargs)
        instance._load_env_dict = load_env_dict
        return instance

    @staticmethod
    def from_name(
        name: str,
        *,
        environment_name: str | None = None,
        required_keys: list[str] = [],
        client: _Client | None = None,
    ) -> "_Secret":
        """Reference a deployed Secret by name.

        Hydration is lazy until the Secret is used.

        Args:
            name: Deployment name of the Secret.
            environment_name: Environment to resolve the name in; defaults to the active environment.
            required_keys: If non-empty, the server asserts these keys exist on the Secret.
            client: Modal client to use for loading; defaults to `Client.from_env()` when omitted.

        Returns:
            A `Secret` handle (possibly not yet hydrated).

        Examples:
            ```python
            secret = modal.Secret.from_name("my-secret")

            @app.function(secrets=[secret])
            def run():
                ...
            ```
        """

        async def _load(self: _Secret, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None):
            req = api_pb2.SecretGetOrCreateRequest(
                deployment_name=name,
                environment_name=load_context.environment_name,
                required_keys=required_keys,
            )
            response = await load_context.client.stub.SecretGetOrCreate(req)
            self._hydrate(response.secret_id, load_context.client, response.metadata)

        rep = _Secret._repr(name, environment_name)
        return _Secret._from_loader(
            _load,
            rep,
            hydrate_lazily=True,
            name=name,
            load_context_overrides=LoadContext(environment_name=environment_name, client=client),
            skip_reload=True,
        )

    @staticmethod
    async def _create_deployed(
        deployment_name: str,
        env_dict: dict[str, str],
        client: _Client | None = None,
        environment_name: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """mdmd:hidden"""

        check_object_name(deployment_name, "Secret")
        if client is None:
            client = await _Client.from_env()
        if overwrite:
            object_creation_type = api_pb2.OBJECT_CREATION_TYPE_CREATE_OVERWRITE_IF_EXISTS
        else:
            object_creation_type = api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS
        request = api_pb2.SecretGetOrCreateRequest(
            deployment_name=deployment_name,
            environment_name=_get_environment_name(environment_name),
            object_creation_type=object_creation_type,
            env_dict=env_dict,
        )
        resp = await client.stub.SecretGetOrCreate(request)
        return resp.secret_id

    @live_method
    async def info(self) -> SecretInfo:
        """Return information about the Secret object."""
        metadata = self._get_metadata()
        creation_info = metadata.creation_info
        return SecretInfo(
            name=metadata.name or None,
            created_at=timestamp_to_localized_dt(creation_info.created_at),
            created_by=creation_info.created_by or None,
        )

    @live_method
    async def update(self, env_dict: dict[str, str]) -> None:
        """Update this Secret, adding or overwriting key-value pairs.

        Like dict.update(), this merges `env_dict` into the existing Secret.
        Keys not mentioned in `env_dict` are left unchanged.
        """
        err = "The `env_dict` argument to `Secret.update` must be a dict[str, str]"
        if not isinstance(env_dict, dict):
            raise InvalidError(err)
        if not all(isinstance(k, str) for k in env_dict.keys()):
            raise InvalidError(err)
        if not all(isinstance(v, str) for v in env_dict.values()):
            raise InvalidError(err)
        updates = [api_pb2.SecretUpdateRequest.Update(key=k, value=v) for k, v in env_dict.items()]
        req = api_pb2.SecretUpdateRequest(secret_id=self.object_id, updates=updates)
        await self._client.stub.SecretUpdate(req)


def _split_env_dict_and_resolvable_secrets(secrets: list[_Secret]) -> tuple[dict[str, str], list[_Secret]]:
    """Split secrets into secrets that can be resolved locally and secrets that are remote.

    Locally resolvable secrets include: `Secret.from_dict`, `Secret.from_dotenv`
    Remote secrets include: `Secret.from_name`
    """
    env_dict: dict[str, str] = {}
    resolvable_secrets: list[_Secret] = []
    for secret in secrets:
        if secret._load_env_dict:
            env_dict |= secret._load_env_dict()
        else:
            resolvable_secrets.append(secret)
    return env_dict, resolvable_secrets


Secret = synchronize_api(_Secret)
