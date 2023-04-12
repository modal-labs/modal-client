# Copyright Modal Labs 2022
import io
import pickle
from typing import Optional

import cloudpickle


from synchronicity.synchronizer import TARGET_INTERFACE_ATTR
from modal_utils import async_utils
from .config import logger
from .exception import InvalidError
import google.protobuf.message
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

        synchronicity_interface = getattr(obj, TARGET_INTERFACE_ATTR, None)
        return obj.object_id, synchronicity_interface


class Unpickler(pickle.Unpickler):
    def __init__(self, client, buf, app=None):
        self.client = client
        self.app = app
        super().__init__(buf)

    def get_proto_for_object(self, object_id) -> Optional[google.protobuf.message.Message]:
        from .functions import _FunctionHandle  # circular dependency :(

        if object_id.startswith(_FunctionHandle._type_prefix + "-"):
            # a bit janky - get the function definition proto from the loaded app objects:
            if self.app is None:
                logger.debug(f"No app provided for deserializing function {object_id}")
                return None

            if object_id not in self.app._object_id_to_object:
                logger.debug(f"Can't find {object_id} in current app, loading without definition")
                return None

            function_handle = self.app._object_id_to_object[object_id]
            if function_handle._function_definition is None:
                logger.debug(f"No function definition was found for {object_id} in provided app - not a container app?")
                return None

            return function_handle._function_definition
        return None

    def persistent_load(self, pid):
        object_id, target_interface = pid
        proto = self.get_proto_for_object(object_id)
        handle = _Handle._from_id(object_id, self.client, proto)

        if target_interface:
            # in case the pickled entity was a wrapped synchronicity object, rewrap it with the same interface:
            return async_utils.synchronizer._translate_out(handle, target_interface)
        return handle


def serialize(obj):
    """Serializes object and replaces all references to the client class by a placeholder."""
    buf = io.BytesIO()
    Pickler(buf).dump(obj)
    return buf.getvalue()


def deserialize(s: bytes, client, container_app=None):
    """Deserializes object and replaces all client placeholders by self."""
    return Unpickler(client, io.BytesIO(s), app=container_app).load()
