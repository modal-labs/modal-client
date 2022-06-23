from modal_proto import api_pb2
from modal_utils.async_utils import retry, synchronize_apis

from .object import Object


class _SharedVolume(Object, type_prefix="sv"):
    def __init__(
        self,
    ):
        super().__init__()

    def _get_creating_message(self):
        return "Creating shared volume..."

    def _get_created_message(self):
        return "Created shared volume."

    async def _load(self, client, app_id, existing_shared_volume_id):
        if existing_shared_volume_id:
            # Volume already exists; do nothing.
            return existing_shared_volume_id

        req = api_pb2.SharedVolumeCreateRequest(app_id=app_id)
        resp = await retry(client.stub.SharedVolumeCreate, base_delay=1)(req)
        return resp.shared_volume_id


SharedVolume, AioSharedVolume = synchronize_apis(_SharedVolume)
