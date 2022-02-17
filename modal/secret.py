from .object import Object
from .proto import api_pb2


class Secret(Object, type_prefix="st"):
    """A dictionary of environment variables for images"""

    @classmethod
    async def create(cls, env_dict={}, app=None):
        app = cls._get_app(app)
        req = api_pb2.SecretCreateRequest(app_id=app.app_id, env_dict=env_dict)
        resp = await app.client.stub.SecretCreate(req)
        return cls._create_object_instance(resp.secret_id, app)
