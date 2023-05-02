# Copyright Modal Labs 2022
from typing import Dict, Optional
from datetime import date

from modal._types import typechecked

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from ._resolver import Resolver
from .exception import InvalidError, deprecation_warning
from .object import _Handle, _Provider


class _SecretHandle(_Handle, type_prefix="st"):
    pass


SecretHandle, AioSecretHandle = synchronize_apis(_SecretHandle)


ENV_DICT_WRONG_TYPE_ERR = "the env_dict argument to Secret has to be a dict[str, str]"


class _Secret(_Provider[_SecretHandle]):
    """Secrets provide a dictionary of environment variables for images.

    Secrets are a secure way to add credentials and other sensitive information
    to the containers your functions run in. You can create and edit secrets on
    [the dashboard](/secrets), or programmatically from Python code.

    See [the secrets guide page](/docs/guide/secrets) for more information.
    """

    @typechecked
    @staticmethod
    def from_dict(
        env_dict: Dict[
            str, str
        ] = {},  # dict of entries to be inserted as environment variables in functions using the secret
        template_type="",  # internal use only
    ):
        """Create a secret from a str-str dictionary.

        Usage:
        ```python
        @stub.function(secret=Secret.from_dict({"FOO": "bar"})
        def run():
            print(os.environ["FOO"])
        ```
        """
        if not isinstance(env_dict, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env_dict.items()
        ):
            raise InvalidError(ENV_DICT_WRONG_TYPE_ERR)

        async def _load(resolver: Resolver, existing_object_id: Optional[str]) -> _SecretHandle:
            req = api_pb2.SecretCreateRequest(
                app_id=resolver.app_id,
                env_dict=env_dict,
                template_type=template_type,
                existing_secret_id=existing_object_id,
            )
            resp = await resolver.client.stub.SecretCreate(req)
            return _SecretHandle._from_id(resp.secret_id, resolver.client, None)

        rep = f"Secret.from_dict([{', '.join(env_dict.keys())}])"
        return _Secret._from_loader(_load, rep)

    def __init__(self, env_dict: Dict[str, str]):
        """`Secret({...})` is deprecated. Please use `Secret.from_dict({...})` instead."""
        deprecation_warning(date(2023, 5, 1), self.__init__.__doc__)
        obj = _Secret.from_dict(env_dict)
        self._init_from_other(obj)

    @staticmethod
    def from_dotenv(dotenv_path=None):  # If provided, location of a .env file
        """Create secrets from a .env file automatically.

        If no argument is provided, it will use the current working directory as the starting
        point for finding a .env file. Note that it does not use the location of the module
        calling .from_dotenv.
        """

        async def _load(resolver: Resolver, existing_object_id: Optional[str]) -> _SecretHandle:
            try:
                from dotenv import find_dotenv, dotenv_values
            except ImportError:
                raise ImportError(
                    "Need the `dotenv` package installed. You can install it by running `pip install python-dotenv`."
                )

            if dotenv_path is not None:
                # TODO(erikbern): this is a path to the .env file, not the directory containing it
                # We should support giving it a directory name and then walking the directory to the root
                _dotenv_path = dotenv_path
            else:
                # TODO(erikbern): dotenv tries to locate .env files based on the location of the file in the stack frame.
                # Since the modal code "intermediates" this, a .env file in the user's local directory won't be picked up.
                # We avoid this by just using the cwd instead.
                _dotenv_path = find_dotenv(usecwd=True)

            env_dict = dotenv_values(_dotenv_path)

            req = api_pb2.SecretCreateRequest(
                app_id=resolver.app_id,
                env_dict=env_dict,
                existing_secret_id=existing_object_id,
            )
            resp = await resolver.client.stub.SecretCreate(req)
            return _SecretHandle._from_id(resp.secret_id, resolver.client, None)

        return _Secret._from_loader(_load, "Secret.from_dotenv()")


Secret, AioSecret = synchronize_apis(_Secret)
