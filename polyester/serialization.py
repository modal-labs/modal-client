import cloudpickle
import pickle

from .config import logger

# TODO: the code below uses the attributes "local_id" and "remote_id" which are not the correct ones
# TODO: deserialization should be done using the Object constructor (setting the client and object_id)


class Pickler(cloudpickle.Pickler):
    def __init__(self, client, type_to_name, buf):
        self.client = client
        self.type_to_name = type_to_name
        super().__init__(buf)

    def persistent_id(self, obj):
        if type(obj) == type(self.client):
            if obj != self.client:
                logger.warn("One client trying to serialize a reference to another client")
            return ("Client", None)
        elif type(obj) in self.type_to_name:
            assert obj._serializable_object_initialized
            class_name = self.type_to_name[type(obj)]
            return (class_name, (obj.local_id, obj.remote_id))


class Unpickler(pickle.Unpickler):
    def __init__(self, client, name_to_type, buf):
        self.client = client
        self.name_to_type = name_to_type
        super().__init__(buf)

    def persistent_load(self, pid):
        type_tag, key_id = pid
        if type_tag == "Client":
            return self.client
        elif type_tag in self.name_to_type:
            cls = self.name_to_type[type_tag]
            local_id, remote_id = key_id
            return cls(client=self.client, local_id=local_id, remote_id=remote_id)
        else:
            raise Exception('unknown type tag "%s" to recover' % type_tag)
