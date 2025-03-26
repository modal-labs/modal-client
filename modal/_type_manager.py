import typing
from typing import Any

import typing_extensions

from modal.exception import InvalidError
from modal_proto import api_pb2


class ParameterProtoSerde(typing.Protocol):
    def encode(self, value: Any, name: typing.Optional[str] = None) -> api_pb2.ClassParameterValue:
        pass

    def decode(self, proto: api_pb2.ClassParameterValue) -> Any:
        pass


class ProtoParameterSerdeRegistry:
    _py_base_type_to_serde: dict[type, ParameterProtoSerde]
    _proto_type_to_serde: dict["api_pb2.ParameterType.ValueType", ParameterProtoSerde]

    def __init__(self):
        self._py_base_type_to_serde = {}
        self._proto_type_to_serde = {}

    def register_encoder(self, python_base_type: type) -> typing.Callable[[ParameterProtoSerde], ParameterProtoSerde]:
        def deco(ph: ParameterProtoSerde) -> ParameterProtoSerde:
            if python_base_type in self._py_base_type_to_serde:
                raise ValueError("Can't register the same encoder type twice")
            self._py_base_type_to_serde[python_base_type] = ph
            return ph

        return deco

    def register_decoder(self, enum_type_value: "api_pb2.ParameterType.ValueType"):
        def deco(ph: ParameterProtoSerde) -> ParameterProtoSerde:
            if enum_type_value in self._proto_type_to_serde:
                raise ValueError("Can't register the same decoder type twice")
            self._proto_type_to_serde[enum_type_value] = ph
            return ph

        return deco

    def get_encoder(self, python_base_type: type) -> ParameterProtoSerde:
        try:
            return self._py_base_type_to_serde[python_base_type]
        except InvalidError:
            raise KeyError(f"No decoder implemented for python type {python_base_type.__name__}")

    def encode(self, python_value: type) -> api_pb2.ClassParameterValue:
        return self.get_encoder(type(python_value)).encode(python_value)

    def get_decoder(self, enum_value: "api_pb2.ParameterType.ValueType") -> ParameterProtoSerde:
        try:
            return self._proto_type_to_serde[enum_value]
        except KeyError:
            try:
                enum_name = api_pb2.ParameterType.Name(enum_value)
            except ValueError:
                enum_name = str(enum_value)

            raise InvalidError(f"No decoder implemented for parameter type {enum_name}")

    def decode(self, param_value: api_pb2.ClassParameterValue) -> Any:
        return self.get_decoder(param_value.type).decode(param_value)

    def base_types_for_serde(self, serde: ParameterProtoSerde) -> set[type]:
        # reverse lookup of python base types that map to a specific serde
        return {t for t, s in self._py_base_type_to_serde.items() if s == serde}

    def supported_base_types(self) -> typing.Collection[type]:
        return self._py_base_type_to_serde.keys()


parameter_serde_registry = ProtoParameterSerdeRegistry()


@parameter_serde_registry.register_encoder(int)
@parameter_serde_registry.register_decoder(api_pb2.PARAM_TYPE_INT)
class IntParameter:
    @staticmethod
    def encode(i: int) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_INT, int_value=i)

    @staticmethod
    def decode(p: api_pb2.ClassParameterValue) -> int:
        return p.int_value


@parameter_serde_registry.register_encoder(str)
@parameter_serde_registry.register_decoder(api_pb2.PARAM_TYPE_STRING)
class StringParameter:
    @staticmethod
    def encode(s: str) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_STRING, string_value=s)

    @staticmethod
    def decode(p: api_pb2.ClassParameterValue) -> str:
        return p.string_value


@parameter_serde_registry.register_encoder(bytes)
@parameter_serde_registry.register_decoder(api_pb2.PARAM_TYPE_BYTES)
class BytesParameter:
    @staticmethod
    def encode(b: bytes) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_BYTES, bytes_value=b)

    @staticmethod
    def decode(p: api_pb2.ClassParameterValue) -> bytes:
        return p.bytes_value


SCHEMA_FACTORY_TYPE = typing.Callable[[type], api_pb2.GenericPayloadType]


class SchemaRegistry:
    _schema_factories: dict[type, SCHEMA_FACTORY_TYPE]

    def __init__(self):
        self._schema_factories = {}

    def add(self, python_base_type: type) -> typing.Callable[[SCHEMA_FACTORY_TYPE], SCHEMA_FACTORY_TYPE]:
        # decorator for schema factory functions for a base type
        def deco(factory_func: SCHEMA_FACTORY_TYPE) -> SCHEMA_FACTORY_TYPE:
            assert python_base_type not in self._schema_factories
            self._schema_factories[python_base_type] = factory_func
            return factory_func

        return deco

    def get(self, python_base_type: type) -> SCHEMA_FACTORY_TYPE:
        try:
            return self._schema_factories[python_base_type]
        except KeyError:
            return unknown_type_schema

    def get_proto_generic_type(self, declared_type: type):
        if origin := typing_extensions.get_origin(declared_type):
            base_type = origin
        else:
            base_type = declared_type

        return self.get(base_type)(declared_type)


schema_registry = SchemaRegistry()


@schema_registry.add(int)
def int_schema(full_python_type: type) -> api_pb2.GenericPayloadType:
    return api_pb2.GenericPayloadType(
        base_type=api_pb2.PARAM_TYPE_INT,
    )


@schema_registry.add(bytes)
def proto_type_def(declared_python_type: type) -> api_pb2.GenericPayloadType:
    return api_pb2.GenericPayloadType(
        base_type=api_pb2.PARAM_TYPE_BYTES,
    )


def unknown_type_schema(declared_python_type: type) -> api_pb2.GenericPayloadType:
    # TODO: add some metadata for unknown types to the type def?
    return api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_UNKNOWN)


@schema_registry.add(str)
def str_schema(full_python_type: type) -> api_pb2.GenericPayloadType:
    return api_pb2.GenericPayloadType(
        base_type=api_pb2.PARAM_TYPE_STRING,
    )


@schema_registry.add(type(None))
def none_type_schema(declared_python_type: type) -> api_pb2.GenericPayloadType:
    return api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_NONE)


@schema_registry.add(list)
def list_schema(full_python_type: type) -> api_pb2.GenericPayloadType:
    args = typing_extensions.get_args(full_python_type)

    return api_pb2.GenericPayloadType(
        base_type=api_pb2.PARAM_TYPE_LIST, sub_types=[schema_registry.get_proto_generic_type(arg) for arg in args]
    )


@schema_registry.add(dict)
def dict_schema(full_python_type: type) -> api_pb2.GenericPayloadType:
    args = typing_extensions.get_args(full_python_type)

    return api_pb2.GenericPayloadType(
        base_type=api_pb2.PARAM_TYPE_DICT,
        sub_types=[schema_registry.get_proto_generic_type(arg_type) for arg_type in args],
    )
