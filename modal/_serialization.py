import pickle

import cloudpickle

from .exception import InvalidError
from .object import Object

PICKLE_PROTOCOL = 4  # Support older Python versions.


class Pickler(cloudpickle.Pickler):
    def __init__(self, app, buf):
        self.app = app
        super().__init__(buf, protocol=PICKLE_PROTOCOL)

    def persistent_id(self, obj):
        if not isinstance(obj, Object):
            return
        if not obj.object_id:
            raise InvalidError(f"Can't serialize object {obj} which hasn't been created.")
        return obj.object_id


class Unpickler(pickle.Unpickler):
    def __init__(self, app, buf):
        self.app = app
        super().__init__(buf)

    def persistent_load(self, pid):
        object_id = pid
        return Object._init_persisted(object_id, self.app)
