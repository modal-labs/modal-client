from .object import Object
from .proto import api_pb2


class Secret(Object, type_prefix="st"):
    """A dictionary of environment variables for images"""

    @classmethod
    async def create(cls, env_dict={}, session=None):
        session = cls._get_session(session)
        req = api_pb2.SecretCreateRequest(app_id=session.session_id, env_dict=env_dict)
        resp = await session.client.stub.SecretCreate(req)
        return cls._create_object_instance(resp.secret_id, session)
