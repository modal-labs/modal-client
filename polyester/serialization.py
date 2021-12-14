import pickle
import uuid

import cloudpickle

from .config import logger
from .object import Object

PICKLE_PROTOCOL = 4  # Support older Python versions.


class Pickler(cloudpickle.Pickler):
    def __init__(self, session, type_to_name, buf):
        self.session = session
        self.type_to_name = type_to_name
        super().__init__(buf, protocol=PICKLE_PROTOCOL)

    def persistent_id(self, obj):
        if not isinstance(obj, Object):
            return
        class_name = self.type_to_name[type(obj)]
        if not obj.object_id:
            raise Exception(f"Can't serialize object {obj} which hasn't been created")
        return (class_name, obj.object_id)


class Unpickler(pickle.Unpickler):
    def __init__(self, session, name_to_type, buf):
        self.session = session
        self.name_to_type = name_to_type
        super().__init__(buf)

    def persistent_load(self, pid):
        type_tag, object_id = pid
        if type_tag not in self.name_to_type:
            raise Exception(f"Unknown tag {type_tag}")
        cls = self.name_to_type[type_tag]
        obj = cls.new(session=self.session)
        obj.set_object_id(object_id, self.session.session_id)
        return obj
