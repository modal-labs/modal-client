# Copyright Modal Labs 2022
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from ._load_context import LoadContext
from .object import Handle, Provider


class _SecretHandle(Handle, type_prefix="st"):
    pass


synchronize_apis(_SecretHandle)


class _Secret(Provider[_SecretHandle]):
    """Secrets provide a dictionary of environment variables for images.

    Secrets are a secure way to add credentials and other sensitive information
    to the containers your functions run in. You can create and edit secrets on
    [the dashboard](/secrets), or programmatically from Python code.

    See [The guide](/docs/guide/secrets) for more information.
    """

    def __init__(self, env_dict={}, template_type=""):
        self._env_dict = env_dict
        self._template_type = template_type
        super().__init__()

    def __repr__(self):
        return f"Secret([{', '.join(self._env_dict.keys())}])"

    async def _load(self, load_context: LoadContext):
        req = api_pb2.SecretCreateRequest(
            app_id=load_context.app_id,
            env_dict=self._env_dict,
            template_type=self._template_type,
            existing_secret_id=load_context.existing_object_id,
        )
        resp = await load_context.client.stub.SecretCreate(req)
        return _SecretHandle(load_context.client, resp.secret_id)


Secret, AioSecret = synchronize_apis(_Secret)
