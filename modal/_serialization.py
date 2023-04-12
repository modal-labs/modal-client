# Copyright Modal Labs 2022
import io
import pickle

import cloudpickle

from synchronicity.synchronizer import TARGET_INTERFACE_ATTR
from modal_utils import async_utils
from .exception import InvalidError
from .object import _Handle, Handle, AioHandle

PICKLE_PROTOCOL = 4  # Support older Python versions.


class Pickler(cloudpickle.Pickler):
    def __init__(self, buf):
        super().__init__(buf, protocol=PICKLE_PROTOCOL)

    def persistent_id(self, obj):
        if not isinstance(obj, (_Handle, Handle, AioHandle)):
            return
        if not obj.object_id:
            raise InvalidError(f"Can't serialize object {obj} which hasn't been created.")
        return obj.object_id, getattr(obj, TARGET_INTERFACE_ATTR, None)


class Unpickler(pickle.Unpickler):
    def __init__(self, client, buf, app=None):
        self.client = client
        self.app = app
        super().__init__(buf)

    def persistent_load(self, pid):
        object_id, target_interface = pid
        if object_id.startswith("fu-"):  # TODO(elias): ugly check
            function_handle = self.app._object_id_to_object[object_id]
            proto = function_handle._function_definition
        else:
            proto = None

        handle = _Handle._from_id(object_id, self.client, proto)
        if target_interface:
            return async_utils.synchronizer._translate_out(handle, target_interface)
        return handle


def serialize(obj):
    """Serializes object and replaces all references to the client class by a placeholder."""
    buf = io.BytesIO()
    Pickler(buf).dump(obj)
    return buf.getvalue()


def deserialize(s: bytes, client, app=None):
    """Deserializes object and replaces all client placeholders by self."""
    return Unpickler(client, io.BytesIO(s), app=app).load()
