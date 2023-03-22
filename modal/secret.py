# Copyright Modal Labs 2022
from typing import Dict

from modal._types import typechecked

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from ._resolver import Resolver
from .object import Handle, Provider


class _SecretHandle(Handle, type_prefix="st"):
    pass


synchronize_apis(_SecretHandle)


class _Secret(Provider[_SecretHandle]):
    """Secrets provide a dictionary of environment variables for images.

    Secrets are a secure way to add credentials and other sensitive information
    to the containers your functions run in. You can create and edit secrets on
    [the dashboard](/secrets), or programmatically from Python code.

    See [the secrets guide page](/docs/guide/secrets) for more information.
    """

    @typechecked
    def __init__(
        self,
        env_dict: Dict[
            str, str
        ] = {},  # dict of entries to be inserted as environment variables in functions using the secret
        template_type="",  # internal use only
    ):
        async def _load(resolver: Resolver, existing_object_id: str) -> _SecretHandle:
            req = api_pb2.SecretCreateRequest(
                app_id=resolver.app_id,
                env_dict=env_dict,
                template_type=template_type,
                existing_secret_id=existing_object_id,
            )
            resp = await resolver.client.stub.SecretCreate(req)
            return _SecretHandle._from_id(resp.secret_id, resolver.client, None)

        rep = f"Secret([{', '.join(env_dict.keys())}])"
        super().__init__(_load, rep)


Secret, AioSecret = synchronize_apis(_Secret)
