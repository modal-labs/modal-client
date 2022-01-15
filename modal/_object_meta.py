from ._async_utils import synchronizer
from ._factory import Factory, make_shared_object_factory_class, make_user_factory
from .config import logger


class ObjectMeta(type):
    type_to_prefix = {}
    prefix_to_type = {}

    def __new__(metacls, name, bases, dct, modal_prefix=None):
        # Synchronize class
        new_cls = synchronizer.create_class(metacls, name, bases, dct)

        # Needed for serialization, also for loading objects dynamically
        if modal_prefix is not None:
            ObjectMeta.type_to_prefix[new_cls] = modal_prefix
            ObjectMeta.prefix_to_type[modal_prefix] = new_cls

        # Create factory class and shared object class
        if not issubclass(new_cls, Factory):
            new_cls._user_factory_class = make_user_factory(new_cls)
            new_cls._shared_object_factory_class = make_shared_object_factory_class(new_cls)

        logger.debug(f"Created Object class {name}")
        return new_cls
