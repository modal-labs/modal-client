from .object import Object
from .proto import api_pb2


class EnvDict(Object):
    def __init__(self, session, env_dict):
        super().__init__(session=session, tag=None)
        self.env_dict = env_dict

    async def _create_impl(self, session):
        req = api_pb2.EnvDictCreateRequest(session_id=session.session_id, env_dict=self.env_dict)
        resp = await session.client.stub.EnvDictCreate(req)
        return resp.env_dict_id
