import cloudpickle
import io
import pickle

from .config import logger


class SerializableObject:
    def __init__(self, client=None):
        self.client = client
        self._serializable_object_initialized = True


class Pickler(cloudpickle.Pickler):
    def __init__(self, client, type_to_name, buf):
        self.client = client
        self.type_to_name = type_to_name
        super().__init__(buf)

    def persistent_id(self, obj):
        if type(obj) == type(self.client):
            if obj != self.client:
                logger.warn('One client trying to serialize a reference to another client')
            return ('Client', None)
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
        if type_tag == 'Client':
            return self.client
        elif type_tag in self.name_to_type:
            cls = self.name_to_type[type_tag]
            local_id, remote_id = key_id
            return cls(client=self.client, local_id=local_id, remote_id=remote_id)
        else:
            raise Exception('unknown type tag "%s" to recover' % type_tag)


class SerializableRegistry:
    def __init__(self):
        self.type_to_name = {}
        self.name_to_type = {}

    def __call__(self, cls):
        '''Class decorator which adds a mixin base class SerializableObject.'''
        cls_name = cls.__name__
        cls_dict = dict(cls.__dict__)
        cls_new = type(cls_name, (cls, SerializableObject), cls_dict)
        logger.debug('Registering class %s as serializable with name %s' % (cls, cls_name))
        self.type_to_name[cls_new] = cls_name
        self.name_to_type[cls_name] = cls_new
        return cls_new

    def serialize(self, client, obj):
        ''' Serializes object and replaces all references to the client class by a placeholder.'''
        buf = io.BytesIO()
        Pickler(client, self.type_to_name, buf).dump(obj)
        return buf.getvalue()

    def deserialize(self, client, s: bytes):
        ''' Deserializes object and replaces all client placeholders by self.'''
        return Unpickler(client, self.name_to_type, io.BytesIO(s)).load()


serializable = SerializableRegistry()
