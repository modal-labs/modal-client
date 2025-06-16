# Copyright Modal Labs 2022
import os
from typing import Optional, Union

from grpclib import GRPCError, Status

from modal_proto import api_pb2

from ._object import _get_environment_name, _Object
from ._resolver import Resolver
from ._runtime.execution_context import is_local
from ._utils.async_utils import synchronize_api
from ._utils.deprecation import deprecation_warning
from ._utils.grpc_utils import retry_transient_errors
from ._utils.name_utils import check_object_name
from .client import _Client
from .exception import InvalidError, NotFoundError

ENV_DICT_WRONG_TYPE_ERR = "the env_dict argument to Secret has to be a dict[str, Union[str, None]]"


class _Secret(_Object, type_prefix="st"):
    """Secrets provide a dictionary of environment variables for images.

    Secrets are a secure way to add credentials and other sensitive information
    to the containers your functions run in. You can create and edit secrets on
    [the dashboard](https://modal.com/secrets), or programmatically from Python code.

    See [the secrets guide page](https://modal.com/docs/guide/secrets) for more information.
    """

    @staticmethod
    def from_dict(
        env_dict: dict[
            str, Union[str, None]
        ] = {},  # dict of entries to be inserted as environment variables in functions using the secret
    ) -> "_Secret":
        """Create a secret from a str-str dictionary. Values can also be `None`, which is ignored.

        Usage:
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

        async def _load(self: _Secret, resolver: Resolver, existing_object_id: Optional[str]):
            if resolver.app_id is not None:
                object_creation_type = api_pb2.OBJECT_CREATION_TYPE_ANONYMOUS_OWNED_BY_APP
            else:
                object_creation_type = api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL

            req = api_pb2.SecretGetOrCreateRequest(
                object_creation_type=object_creation_type,
                env_dict=env_dict_filtered,
                app_id=resolver.app_id,
                environment_name=resolver.environment_name,
            )
            try:
                resp = await resolver.client.stub.SecretGetOrCreate(req)
            except GRPCError as exc:
                if exc.status == Status.INVALID_ARGUMENT:
                    raise InvalidError(exc.message)
                if exc.status == Status.FAILED_PRECONDITION:
                    raise InvalidError(exc.message)
                raise
            self._hydrate(resp.secret_id, resolver.client, None)

        rep = f"Secret.from_dict([{', '.join(env_dict.keys())}])"
        return _Secret._from_loader(_load, rep, hydrate_lazily=True)

    @staticmethod
    def from_local_environ(
        env_keys: list[str],  # list of local env vars to be included for remote execution
    ) -> "_Secret":
        """Create secrets from local environment variables automatically."""

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
    def from_dotenv(path=None, *, filename=".env") -> "_Secret":
        """Create secrets from a .env file automatically.

        If no argument is provided, it will use the current working directory as the starting
        point for finding a `.env` file. Note that it does not use the location of the module
        calling `Secret.from_dotenv`.

        If called with an argument, it will use that as a starting point for finding `.env` files.
        In particular, you can call it like this:
        ```python
        @app.function(secrets=[modal.Secret.from_dotenv(__file__)])
        def run():
            print(os.environ["USERNAME"])  # Assumes USERNAME is defined in your .env file
        ```

        This will use the location of the script calling `modal.Secret.from_dotenv` as a
        starting point for finding the `.env` file.

        A file named `.env` is expected by default, but this can be overridden with the `filename`
        keyword argument:

        ```python
        @app.function(secrets=[modal.Secret.from_dotenv(filename=".env-dev")])
        def run():
            ...
        ```
        """

        async def _load(self: _Secret, resolver: Resolver, existing_object_id: Optional[str]):
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

            env_dict = dotenv_values(dotenv_path)

            req = api_pb2.SecretGetOrCreateRequest(
                object_creation_type=api_pb2.OBJECT_CREATION_TYPE_ANONYMOUS_OWNED_BY_APP,
                env_dict=env_dict,
                app_id=resolver.app_id,
            )
            resp = await resolver.client.stub.SecretGetOrCreate(req)

            self._hydrate(resp.secret_id, resolver.client, None)

        return _Secret._from_loader(_load, "Secret.from_dotenv()", hydrate_lazily=True)

    @staticmethod
    def from_name(
        name: str,
        *,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        required_keys: list[
            str
        ] = [],  # Optionally, a list of required environment variables (will be asserted server-side)
    ) -> "_Secret":
        """Reference a Secret by its name.

        In contrast to most other Modal objects, named Secrets must be provisioned
        from the Dashboard. See other methods for alternate ways of creating a new
        Secret from code.

        ```python
        secret = modal.Secret.from_name("my-secret")

        @app.function(secrets=[secret])
        def run():
           ...
        ```
        """

        async def _load(self: _Secret, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.SecretGetOrCreateRequest(
                deployment_name=name,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
                required_keys=required_keys,
            )
            try:
                response = await resolver.client.stub.SecretGetOrCreate(req)
            except GRPCError as exc:
                if exc.status == Status.NOT_FOUND:
                    raise NotFoundError(exc.message)
                else:
                    raise
            self._hydrate(response.secret_id, resolver.client, None)

        return _Secret._from_loader(_load, "Secret()", hydrate_lazily=True)

    @staticmethod
    async def lookup(
        name: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        required_keys: list[str] = [],
    ) -> "_Secret":
        """mdmd:hidden"""
        deprecation_warning(
            (2025, 1, 27),
            "`modal.Secret.lookup` is deprecated and will be removed in a future release."
            " It can be replaced with `modal.Secret.from_name`."
            "\n\nSee https://modal.com/docs/guide/modal-1-0-migration for more information.",
        )
        obj = _Secret.from_name(
            name, namespace=namespace, environment_name=environment_name, required_keys=required_keys
        )
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client)
        await resolver.load(obj)
        return obj

    @staticmethod
    async def create_deployed(
        deployment_name: str,
        env_dict: dict[str, str],
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
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
            namespace=namespace,
            environment_name=_get_environment_name(environment_name),
            object_creation_type=object_creation_type,
            env_dict=env_dict,
        )
        resp = await retry_transient_errors(client.stub.SecretGetOrCreate, request)
        return resp.secret_id


Secret = synchronize_api(_Secret)
