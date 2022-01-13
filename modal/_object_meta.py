from ._async_utils import synchronizer
from ._factory import BaseFactory, make_factory, make_shared_object_factory_class
from .config import logger


class ObjectMeta(type):
    type_to_name = {}
    name_to_type = {}

    def __new__(metacls, name, bases, dct):
        # Synchronize class
        new_cls = synchronizer.create_class(metacls, name, bases, dct)

        # Register class as serializable
        ObjectMeta.type_to_name[new_cls] = name
        ObjectMeta.name_to_type[name] = new_cls

        # Create factory class and shared object class
        if not issubclass(new_cls, BaseFactory):
            new_cls._factory_class = make_factory(new_cls)
            new_cls._shared_object_factory_class = make_shared_object_factory_class(new_cls)

        logger.debug(f"Created Object class {name}")
        return new_cls
