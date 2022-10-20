# Copyright Modal Labs 2022
import io
import pickle

import cloudpickle

from .exception import InvalidError
from .object import Handle

PICKLE_PROTOCOL = 4  # Support older Python versions.


class Pickler(cloudpickle.Pickler):
    def __init__(self, buf):
        super().__init__(buf, protocol=PICKLE_PROTOCOL)

    def persistent_id(self, obj):
        if not isinstance(obj, Handle):
            return
        if not obj.object_id:
            raise InvalidError(f"Can't serialize object {obj} which hasn't been created.")
        return obj.object_id


class Unpickler(pickle.Unpickler):
    def __init__(self, client, buf):
        self.client = client
        super().__init__(buf)

    def persistent_load(self, pid):
        object_id = pid
        return Handle._from_id(object_id, self.client)


def serialize(obj):
    """Serializes object and replaces all references to the client class by a placeholder."""
    buf = io.BytesIO()
    Pickler(buf).dump(obj)
    return buf.getvalue()


def deserialize(s: bytes, client):
    """Deserializes object and replaces all client placeholders by self."""
    return Unpickler(client, io.BytesIO(s)).load()
