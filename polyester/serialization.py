import pickle

import cloudpickle

from .config import logger
from .object import Object

# TODO: the code below uses the attributes "local_id" and "remote_id" which are not the correct ones
# TODO: deserialization should be done using the Object constructor (setting the client and object_id)


class Pickler(cloudpickle.Pickler):
    def __init__(self, client, type_to_name, buf):
        self.client = client
        self.type_to_name = type_to_name
        super().__init__(buf)

    def persistent_id(self, obj):
        if not isinstance(obj, Object):
            return
        if not obj.created:
            # TODO: in the future, if the object is a reference to a persisted object,
            # we should probably permit it to be serialized too
            raise Exception("Can only serialize objects that have been created")
        class_name = self.type_to_name[type(obj)]
        return (class_name, obj.object_id)


class Unpickler(pickle.Unpickler):
    def __init__(self, client, name_to_type, buf):
        self.client = client
        self.name_to_type = name_to_type
        super().__init__(buf)

    def persistent_load(self, pid):
        type_tag, object_id = pid
        if type_tag not in self.name_to_type:
            raise Exception(f"Unknown tag {type_tag}")
        cls = self.name_to_type[type_tag]
        obj = cls()
        obj.client = self.client  # TODO: set session as well
        obj.create_from_id(object_id)
        return obj
