# Copyright Modal Labs 2022
import io
import pickle
from typing import Optional

import cloudpickle
from synchronicity import Interface
from synchronicity.synchronizer import TARGET_INTERFACE_ATTR

from modal_utils import async_utils

from .exception import InvalidError
from .object import Handle, _Handle

PICKLE_PROTOCOL = 4  # Support older Python versions.


def get_synchronicity_interface(obj) -> Optional[Interface]:
    return getattr(obj, TARGET_INTERFACE_ATTR, None)


def restore_synchronicity_interface(raw_obj, target_interface: Optional[Interface]):
    if target_interface:
        return async_utils.synchronizer._translate_out(raw_obj, target_interface)
    return raw_obj


class Pickler(cloudpickle.Pickler):
    def __init__(self, buf):
        super().__init__(buf, protocol=PICKLE_PROTOCOL)

    def persistent_id(self, obj):
        if not isinstance(obj, (_Handle, Handle)):
            return
        if not obj.object_id:
            raise InvalidError(f"Can't serialize object {obj} which hasn't been created.")
        return (obj.object_id, get_synchronicity_interface(obj), obj._get_metadata())


class Unpickler(pickle.Unpickler):
    def __init__(self, client, buf):
        self.client = client
        super().__init__(buf)

    def persistent_load(self, pid):
        (object_id, target_interface, handle_proto) = pid
        raw_obj = _Handle._new_hydrated(object_id, self.client, handle_proto)
        return restore_synchronicity_interface(raw_obj, target_interface)


def serialize(obj):
    """Serializes object and replaces all references to the client class by a placeholder."""
    buf = io.BytesIO()
    Pickler(buf).dump(obj)
    return buf.getvalue()


def deserialize(s: bytes, client):
    """Deserializes object and replaces all client placeholders by self."""
    return Unpickler(client, io.BytesIO(s)).load()
