# Copyright Modal Labs 2022
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from .exception import NotFoundError

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

    def __init__(self, env_dict={}, template_type=""):
        async def _load(resolver: Resolver) -> _SecretHandle:
            req = api_pb2.SecretCreateRequest(
                app_id=resolver.app_id,
                env_dict=env_dict,
                template_type=template_type,
                existing_secret_id=resolver.existing_object_id,
            )
            resp = await resolver.client.stub.SecretCreate(req)
            return _SecretHandle._from_id(resp.secret_id, resolver.client, None)

        rep = f"Secret([{', '.join(env_dict.keys())}])"
        super().__init__(_load, rep)


async def _resolve_secret(resolver: Resolver, secret: _Secret) -> str:
    """mdmd:hidden"""
    try:
        secret_id = await resolver.load(secret)
    except NotFoundError as ex:
        if isinstance(secret, _Secret):
            msg = f"Secret {secret} was not found"
        else:
            msg = str(ex)
        msg += ". You can add secrets to your account at https://modal.com/secrets"
        raise NotFoundError(msg)

    return secret_id


Secret, AioSecret = synchronize_apis(_Secret)
