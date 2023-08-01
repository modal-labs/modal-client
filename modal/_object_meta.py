# Copyright Modal Labs 2022
from typing import Any, Dict


class HandleMeta(type):
    # TODO(erikbern): delete this one soon
    prefix_to_type: Dict[str, Any] = {}

    def __new__(metacls, name, bases, dct, type_prefix=None):
        # Set the _type_prefix variable
        dct.update(dict(_type_prefix=type_prefix))

        # Create class
        new_cls = type.__new__(metacls, name, bases, dct)

        # Needed for serialization, also for loading objects dynamically
        if type_prefix is not None:
            metacls.prefix_to_type[type_prefix] = new_cls

        return new_cls


class ProviderMeta(type):
    prefix_to_type: Dict[str, Any] = {}

    def __new__(metacls, name, bases, dct, type_prefix=None):
        # Set the _type_prefix variable
        dct.update(dict(_type_prefix=type_prefix))

        # Create class
        new_cls = type.__new__(metacls, name, bases, dct)

        # Needed for serialization, also for loading objects dynamically
        if type_prefix is not None:
            metacls.prefix_to_type[type_prefix] = new_cls

        return new_cls
