import abc
import typing
from abc import abstractmethod
from typing import Any

import typing_extensions

from modal.exception import InvalidError
from modal_proto import api_pb2
from modal_version import __version__ as client_version

PH = typing.TypeVar("PH", bound=type["TypeManager"])


class TypeRegistry:
    """Various type-related conversion logic, grouped by Type

    Mainly used for class parameter schemas + serialization today, but we could extend it to also more
    cleanly incorporate the `modal run` CLI parsing logic and other potential things.
    """

    _py_base_type_to_handler: dict[type, "TypeManager"]
    _proto_type_to_handler: dict["api_pb2.ParameterType.ValueType", "TypeManager"]

    def __init__(self):
        self._py_base_type_to_handler = {}
        self._proto_type_to_handler = {}

    def register_encoder(self, python_base_type: type) -> typing.Callable[[PH], PH]:
        def deco(ph: type[TypeManager]):
            if python_base_type in self._py_base_type_to_handler:
                raise ValueError("Can't register the same encoder type twice")
            self._py_base_type_to_handler[python_base_type] = ph.singleton()
            return ph

        return deco

    def register_decoder(self, enum_type_value: "api_pb2.ParameterType.ValueType"):
        def deco(ph: type[TypeManager]):
            if enum_type_value in self._proto_type_to_handler:
                raise ValueError("Can't register the same decoder type twice")
            self._proto_type_to_handler[enum_type_value] = ph.singleton()
            return ph

        return deco

    def _for_base_type(self, python_base_type: type) -> "TypeManager":
        # private helper method
        if handler := self._py_base_type_to_handler.get(python_base_type):
            return handler

        # use UnknownType for unknown types
        return UnknownTypeHandler.singleton()

    def _for_declared_type(self, python_declared_type: type) -> "TypeManager":
        """TypeManager for a type annotation (that could be a generic type)"""
        if origin := typing_extensions.get_origin(python_declared_type):
            base_type = origin  # look up by generic type, not exact type
        else:
            base_type = python_declared_type

        return self._for_base_type(base_type)

    def get_proto_generic_type(self, python_declared_type: type) -> api_pb2.GenericPayloadType:
        # Convenience method
        return self._for_declared_type(python_declared_type).proto_type_def(python_declared_type)

    def for_runtime_value(self, python_runtime_value: Any) -> "TypeManager":
        return self._for_base_type(type(python_runtime_value))

    def for_proto_enum(self, proto_type_enum: "api_pb2.ParameterType.ValueType") -> "TypeManager":
        if handler := self._proto_type_to_handler.get(proto_type_enum):
            return handler

        if proto_type_enum not in api_pb2.ParameterType.values():
            raise InvalidError(
                f"modal {client_version} doesn't recognize payload type {proto_type_enum}. "
                f"Try upgrading to a later version of Modal."
            )

        enum_name = api_pb2.ParameterType.Name(proto_type_enum)
        raise InvalidError(f"No payload handler implemented for payload type {enum_name}")

    def base_types_for_handler(self, payload_handler: "TypeManager") -> set[type]:
        # reverse lookup of which types map to a specific handler
        return {k for k, h in self._py_base_type_to_handler.items() if h == payload_handler}


class TypeManager(metaclass=abc.ABCMeta):
    allow_as_class_parameter = False
    _singleton: typing.ClassVar[typing.Optional[typing_extensions.Self]] = None

    @abstractmethod
    def proto_type_def(self, declared_python_type: type) -> api_pb2.GenericPayloadType:
        ...
        # This should return the protobuf representation of the python type declaration

    def to_class_parameter_value(self, python_value: Any) -> api_pb2.ClassParameterValue:
        # Only needs to be implemented for types supporting class parameters
        raise NotImplementedError(f"to_class_parameter_value not implemented for {self.__class__.__name__}")

    def from_class_parameter_value(self, struct: api_pb2.ClassParameterValue) -> Any:
        # Only needs to be implemented for types supporting class parameters
        raise NotImplementedError(f"from_class_parameter_value not implemented for {self.__class__.__name__}")

    @classmethod
    def singleton(cls) -> typing_extensions.Self:
        # a bit hacky, but this lets us register the same manager instance multiple times
        # with chained decorators...
        if "_singleton" not in cls.__dict__:  # uses .__dict__ to avoid parent class resolution of getattr
            cls._singleton = cls()
        return cls._singleton


type_registry = TypeRegistry()


@type_registry.register_encoder(int)
@type_registry.register_decoder(api_pb2.PARAM_TYPE_INT)
class IntType(TypeManager):
    allow_as_class_parameter = True

    def to_class_parameter_value(self, i: int) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_INT, int_value=i)

    def from_class_parameter_value(self, p: api_pb2.ClassParameterValue) -> int:
        return p.int_value

    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        return api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_INT,
        )


@type_registry.register_encoder(str)
@type_registry.register_decoder(api_pb2.PARAM_TYPE_STRING)
class StringType(TypeManager):
    allow_as_class_parameter = True

    def to_class_parameter_value(self, s: str) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_STRING, string_value=s)

    def from_class_parameter_value(self, p: api_pb2.ClassParameterValue) -> str:
        return p.string_value

    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        return api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_STRING,
        )


@type_registry.register_encoder(bytes)
@type_registry.register_decoder(api_pb2.PARAM_TYPE_BYTES)
class BytesType(TypeManager):
    allow_as_class_parameter = True

    def to_class_parameter_value(self, b: bytes) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_BYTES, bytes_value=b)

    def from_class_parameter_value(self, p: api_pb2.ClassParameterValue) -> bytes:
        return p.bytes_value

    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        return api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_BYTES,
        )


class UnknownTypeHandler(TypeManager):
    # undecorated fields or those decorated with unrecognized types
    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        return api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_UNKNOWN)


@type_registry.register_encoder(type(None))
@type_registry.register_decoder(api_pb2.PARAM_TYPE_NONE)
class NoneTypeHandler(TypeManager):
    def proto_type_def(self, declared_python_type: type) -> api_pb2.GenericPayloadType:
        return api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_NONE)


@type_registry.register_encoder(list)  # might want to do tuple separately
@type_registry.register_decoder(api_pb2.PARAM_TYPE_LIST)
class ListType(TypeManager):
    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        args = typing_extensions.get_args(full_python_type)

        return api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_LIST, sub_types=[type_registry.get_proto_generic_type(arg) for arg in args]
        )


@type_registry.register_encoder(dict)
@type_registry.register_decoder(api_pb2.PARAM_TYPE_DICT)
class DictType(TypeManager):
    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        args = typing_extensions.get_args(full_python_type)

        return api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_DICT,
            sub_types=[type_registry.get_proto_generic_type(arg_type) for arg_type in args],
        )
