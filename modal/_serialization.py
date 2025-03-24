# Copyright Modal Labs 2022
import abc
import inspect
import io
import pickle
import typing
from abc import abstractmethod
from inspect import Parameter
from typing import Any

import typing_extensions

from modal._utils.async_utils import synchronizer
from modal_proto import api_pb2
from modal_version import __version__ as client_version

from ._object import _Object
from ._vendor import cloudpickle
from .config import logger
from .exception import DeserializationError, ExecutionError, InvalidError
from .object import Object

if typing.TYPE_CHECKING:
    import modal.client

PICKLE_PROTOCOL = 4  # Support older Python versions.


class Pickler(cloudpickle.Pickler):
    def __init__(self, buf):
        super().__init__(buf, protocol=PICKLE_PROTOCOL)

    def persistent_id(self, obj):
        from modal.partial_function import PartialFunction

        if isinstance(obj, _Object):
            flag = "_o"
        elif isinstance(obj, Object):
            flag = "o"
        elif isinstance(obj, PartialFunction):
            # Special case for PartialObject since it's a synchronicity wrapped object
            # that's set on serialized classes.
            # The resulting pickled instance can't be deserialized without this in a
            # new process, since the original referenced synchronizer will have different
            # values for `._original_attr` etc.

            impl_object = synchronizer._translate_in(obj)
            attributes = impl_object.__dict__.copy()
            # ugly - we remove the `._wrapped_attr` attribute from the implementation instance
            # to avoid referencing and therefore pickling the wrapped instance despite having
            # translated it to the implementation type

            # it would be nice if we could avoid this by not having the wrapped instances
            # be directly linked from objects and instead having a lookup table in the Synchronizer:
            if synchronizer._wrapped_attr and synchronizer._wrapped_attr in attributes:
                attributes.pop(synchronizer._wrapped_attr)

            return ("sync", (impl_object.__class__, attributes))
        else:
            return
        if not obj.is_hydrated:
            raise InvalidError(f"Can't serialize object {obj} which hasn't been hydrated.")
        return (obj.object_id, flag, obj._get_metadata())


class Unpickler(pickle.Unpickler):
    def __init__(self, client, buf):
        self.client = client
        super().__init__(buf)

    def persistent_load(self, pid):
        if len(pid) == 2:
            # more general protocol
            obj_type, obj_data = pid
            if obj_type == "sync":  # synchronicity wrapped object
                # not actually a proto object in this case but the underlying object of a synchronicity object
                impl_class, attributes = obj_data
                impl_instance = impl_class.__new__(impl_class)
                impl_instance.__dict__.update(attributes)
                return synchronizer._translate_out(impl_instance)
            else:
                raise ExecutionError("Unknown serialization format")

        # old protocol, always a 3-tuple
        (object_id, flag, handle_proto) = pid
        if flag in ("o", "p", "h"):
            return Object._new_hydrated(object_id, self.client, handle_proto)
        elif flag in ("_o", "_p", "_h"):
            return _Object._new_hydrated(object_id, self.client, handle_proto)
        else:
            raise InvalidError("bad flag")


def serialize(obj: Any) -> bytes:
    """Serializes object and replaces all references to the client class by a placeholder."""
    buf = io.BytesIO()
    Pickler(buf).dump(obj)
    return buf.getvalue()


def deserialize(s: bytes, client) -> Any:
    """Deserializes object and replaces all client placeholders by self."""
    from ._runtime.execution_context import is_local  # Avoid circular import

    env = "local" if is_local() else "remote"
    try:
        return Unpickler(client, io.BytesIO(s)).load()
    except AttributeError as exc:
        # We use a different cloudpickle version pre- and post-3.11. Unfortunately cloudpickle
        # doesn't expose some kind of serialization version number, so we have to guess based
        # on the error message.
        if "Can't get attribute '_make_function'" in str(exc):
            raise DeserializationError(
                "Deserialization failed due to a version mismatch between local and remote environments. "
                "Try changing the Python version in your Modal image to match your local Python version. "
            ) from exc
        else:
            # On Python 3.10+, AttributeError has `.name` and `.obj` attributes for better custom reporting
            raise DeserializationError(
                f"Deserialization failed with an AttributeError, {exc}. This is probably because"
                " you have different versions of a library in your local and remote environments."
            ) from exc
    except ModuleNotFoundError as exc:
        raise DeserializationError(
            f"Deserialization failed because the '{exc.name}' module is not available in the {env} environment."
        ) from exc
    except Exception as exc:
        if env == "remote":
            # We currently don't always package the full traceback from errors in the remote entrypoint logic.
            # So try to include as much information as we can in the main error message.
            more = f": {type(exc)}({str(exc)})"
        else:
            # When running locally, we can just rely on standard exception chaining.
            more = " (see above for details)"
        raise DeserializationError(
            f"Encountered an error when deserializing an object in the {env} environment{more}."
        ) from exc


def _serialize_asgi(obj: Any) -> api_pb2.Asgi:
    def flatten_headers(obj):
        return [s for k, v in obj for s in (k, v)]

    if obj is None:
        return api_pb2.Asgi()

    msg_type = obj.get("type")

    if msg_type == "http":
        return api_pb2.Asgi(
            http=api_pb2.Asgi.Http(
                http_version=obj.get("http_version", "1.1"),
                method=obj["method"],
                scheme=obj.get("scheme", "http"),
                path=obj["path"],
                query_string=obj.get("query_string"),
                headers=flatten_headers(obj.get("headers", [])),
                client_host=obj["client"][0] if obj.get("client") else None,
                client_port=obj["client"][1] if obj.get("client") else None,
            )
        )
    elif msg_type == "http.request":
        return api_pb2.Asgi(
            http_request=api_pb2.Asgi.HttpRequest(
                body=obj.get("body"),
                more_body=obj.get("more_body"),
            )
        )
    elif msg_type == "http.response.start":
        return api_pb2.Asgi(
            http_response_start=api_pb2.Asgi.HttpResponseStart(
                status=obj["status"],
                headers=flatten_headers(obj.get("headers", [])),
                trailers=obj.get("trailers"),
            )
        )
    elif msg_type == "http.response.body":
        return api_pb2.Asgi(
            http_response_body=api_pb2.Asgi.HttpResponseBody(
                body=obj.get("body"),
                more_body=obj.get("more_body"),
            )
        )
    elif msg_type == "http.response.trailers":
        return api_pb2.Asgi(
            http_response_trailers=api_pb2.Asgi.HttpResponseTrailers(
                headers=flatten_headers(obj.get("headers", [])),
                more_trailers=obj.get("more_trailers"),
            )
        )
    elif msg_type == "http.disconnect":
        return api_pb2.Asgi(http_disconnect=api_pb2.Asgi.HttpDisconnect())

    elif msg_type == "websocket":
        return api_pb2.Asgi(
            websocket=api_pb2.Asgi.Websocket(
                http_version=obj.get("http_version", "1.1"),
                scheme=obj.get("scheme", "ws"),
                path=obj["path"],
                query_string=obj.get("query_string"),
                headers=flatten_headers(obj.get("headers", [])),
                client_host=obj["client"][0] if obj.get("client") else None,
                client_port=obj["client"][1] if obj.get("client") else None,
                subprotocols=obj.get("subprotocols"),
            )
        )
    elif msg_type == "websocket.connect":
        return api_pb2.Asgi(
            websocket_connect=api_pb2.Asgi.WebsocketConnect(),
        )
    elif msg_type == "websocket.accept":
        return api_pb2.Asgi(
            websocket_accept=api_pb2.Asgi.WebsocketAccept(
                subprotocol=obj.get("subprotocol"),
                headers=flatten_headers(obj.get("headers", [])),
            )
        )
    elif msg_type == "websocket.receive":
        return api_pb2.Asgi(
            websocket_receive=api_pb2.Asgi.WebsocketReceive(
                bytes=obj.get("bytes"),
                text=obj.get("text"),
            )
        )
    elif msg_type == "websocket.send":
        return api_pb2.Asgi(
            websocket_send=api_pb2.Asgi.WebsocketSend(
                bytes=obj.get("bytes"),
                text=obj.get("text"),
            )
        )
    elif msg_type == "websocket.disconnect":
        return api_pb2.Asgi(
            websocket_disconnect=api_pb2.Asgi.WebsocketDisconnect(
                code=obj.get("code"),
            )
        )
    elif msg_type == "websocket.close":
        return api_pb2.Asgi(
            websocket_close=api_pb2.Asgi.WebsocketClose(
                code=obj.get("code"),
                reason=obj.get("reason"),
            )
        )

    else:
        logger.debug("skipping serialization of unknown ASGI message type %r", msg_type)
        return api_pb2.Asgi()


def _deserialize_asgi(asgi: api_pb2.Asgi) -> Any:
    def unflatten_headers(obj):
        return list(zip(obj[::2], obj[1::2]))

    msg_type = asgi.WhichOneof("type")

    if msg_type == "http":
        return {
            "type": "http",
            "http_version": asgi.http.http_version,
            "method": asgi.http.method,
            "scheme": asgi.http.scheme,
            "path": asgi.http.path,
            "query_string": asgi.http.query_string,
            "headers": unflatten_headers(asgi.http.headers),
            **({"client": (asgi.http.client_host, asgi.http.client_port)} if asgi.http.HasField("client_host") else {}),
            "extensions": {
                "http.response.trailers": {},
            },
        }
    elif msg_type == "http_request":
        return {
            "type": "http.request",
            "body": asgi.http_request.body,
            "more_body": asgi.http_request.more_body,
        }
    elif msg_type == "http_response_start":
        return {
            "type": "http.response.start",
            "status": asgi.http_response_start.status,
            "headers": unflatten_headers(asgi.http_response_start.headers),
            "trailers": asgi.http_response_start.trailers,
        }
    elif msg_type == "http_response_body":
        return {
            "type": "http.response.body",
            "body": asgi.http_response_body.body,
            "more_body": asgi.http_response_body.more_body,
        }
    elif msg_type == "http_response_trailers":
        return {
            "type": "http.response.trailers",
            "headers": unflatten_headers(asgi.http_response_trailers.headers),
            "more_trailers": asgi.http_response_trailers.more_trailers,
        }
    elif msg_type == "http_disconnect":
        return {"type": "http.disconnect"}

    elif msg_type == "websocket":
        return {
            "type": "websocket",
            "http_version": asgi.websocket.http_version,
            "scheme": asgi.websocket.scheme,
            "path": asgi.websocket.path,
            "query_string": asgi.websocket.query_string,
            "headers": unflatten_headers(asgi.websocket.headers),
            **(
                {"client": (asgi.websocket.client_host, asgi.websocket.client_port)}
                if asgi.websocket.HasField("client_host")
                else {}
            ),
            "subprotocols": list(asgi.websocket.subprotocols),
        }
    elif msg_type == "websocket_connect":
        return {"type": "websocket.connect"}
    elif msg_type == "websocket_accept":
        return {
            "type": "websocket.accept",
            "subprotocol": asgi.websocket_accept.subprotocol if asgi.websocket_accept.HasField("subprotocol") else None,
            "headers": unflatten_headers(asgi.websocket_accept.headers),
        }
    elif msg_type == "websocket_receive":
        return {
            "type": "websocket.receive",
            "bytes": asgi.websocket_receive.bytes if asgi.websocket_receive.HasField("bytes") else None,
            "text": asgi.websocket_receive.text if asgi.websocket_receive.HasField("text") else None,
        }
    elif msg_type == "websocket_send":
        return {
            "type": "websocket.send",
            "bytes": asgi.websocket_send.bytes if asgi.websocket_send.HasField("bytes") else None,
            "text": asgi.websocket_send.text if asgi.websocket_send.HasField("text") else None,
        }
    elif msg_type == "websocket_disconnect":
        return {
            "type": "websocket.disconnect",
            "code": asgi.websocket_disconnect.code if asgi.websocket_disconnect.HasField("code") else 1005,
        }
    elif msg_type == "websocket_close":
        return {
            "type": "websocket.close",
            "code": asgi.websocket_close.code if asgi.websocket_close.HasField("code") else 1000,
            "reason": asgi.websocket_close.reason,
        }

    else:
        assert msg_type is None
        return None


def serialize_data_format(obj: Any, data_format: int) -> bytes:
    """Similar to serialize(), but supports other data formats."""
    if data_format == api_pb2.DATA_FORMAT_PICKLE:
        return serialize(obj)
    elif data_format == api_pb2.DATA_FORMAT_PROTO:
        args, kwargs = obj
        return encode_proto_payload(args, kwargs).SerializeToString()
    elif data_format == api_pb2.DATA_FORMAT_ASGI:
        return _serialize_asgi(obj).SerializeToString(deterministic=True)
    elif data_format == api_pb2.DATA_FORMAT_GENERATOR_DONE:
        assert isinstance(obj, api_pb2.GeneratorDone)
        return obj.SerializeToString(deterministic=True)
    else:
        raise InvalidError(f"Unknown data format {data_format!r}")


def encode_proto_payload(args: tuple, kwargs: dict[str, Any]) -> api_pb2.Payload:
    list_args = list(args)
    args_value = type_register.get_encoder(list_args).encode(list_args)
    kwargs_value = type_register.get_encoder(kwargs).encode(kwargs)
    return api_pb2.Payload(
        args=args_value.list_value,
        kwargs=kwargs_value.dict_value,
    )


def decode_proto_payload(payload: api_pb2.Payload) -> tuple[typing.Sequence[Any], dict[str, Any]]:
    args = ListType()._decode_list_value(payload.args)
    kwargs = DictType()._decode_dict_value(payload.kwargs)
    return (args, kwargs)


def deserialize_data_format(s: bytes, data_format: int, client) -> Any:
    if data_format == api_pb2.DATA_FORMAT_PICKLE:
        return deserialize(s, client)
    if data_format == api_pb2.DATA_FORMAT_PROTO:
        return
    elif data_format == api_pb2.DATA_FORMAT_ASGI:
        return _deserialize_asgi(api_pb2.Asgi.FromString(s))
    elif data_format == api_pb2.DATA_FORMAT_GENERATOR_DONE:
        return api_pb2.GeneratorDone.FromString(s)
    else:
        raise InvalidError(f"Unknown data format {data_format!r}")


class ClsConstructorPickler(pickle.Pickler):
    def __init__(self, buf):
        super().__init__(buf, protocol=PICKLE_PROTOCOL)

    def persistent_id(self, obj):
        if isinstance(obj, (_Object, Object)):
            if not obj.object_id:
                raise InvalidError(f"Can't serialize object {obj} which hasn't been created.")
            return True


def check_valid_cls_constructor_arg(key, obj):
    # Basically pickle, but with support for modal objects
    buf = io.BytesIO()
    try:
        ClsConstructorPickler(buf).dump(obj)
        return True
    except (AttributeError, ValueError):
        raise ValueError(
            f"Only pickle-able types are allowed in remote class constructors: argument {key} of type {type(obj)}."
        )


def apply_defaults(
    python_params: typing.Mapping[str, Any], schema: typing.Sequence[api_pb2.ClassParameterSpec]
) -> dict[str, Any]:
    """Apply any declared defaults from the provided schema, if values aren't provided in python_params

    Conceptually similar to inspect.BoundArguments.apply_defaults.

    Note: Apply this before serializing parameters in order to get consistent parameter
        pools regardless if a value is explicitly provided or not.
    """
    result = {**python_params}
    for schema_param in schema:
        if schema_param.has_default and schema_param.name not in python_params:
            default_field_name = schema_param.WhichOneof("default_oneof")
            if default_field_name is None:
                raise InvalidError(f"{schema_param.name} declared as having a default, but has no default value")
            result[schema_param.name] = getattr(schema_param, default_field_name)
    return result


def encode_parameter_value(name: str, python_value: Any) -> api_pb2.ClassParameterValue:
    """Map to proto payload struct using python runtime type information"""
    payload_handler = type_register.get_encoder(python_value)
    assert payload_handler.allow_as_class_parameter  # this should have been tested for earlier
    struct = payload_handler.encode(python_value)
    struct.name = name
    return struct


def serialize_proto_params(python_params: dict[str, Any]) -> bytes:
    proto_params: list[api_pb2.ClassParameterValue] = []
    for param_name, python_value in python_params.items():
        proto_params.append(encode_parameter_value(param_name, python_value))
    proto_bytes = api_pb2.ClassParameterSet(parameters=proto_params).SerializeToString(deterministic=True)
    return proto_bytes


def deserialize_proto_params(serialized_params: bytes) -> dict[str, Any]:
    proto_struct = api_pb2.ClassParameterSet()
    proto_struct.ParseFromString(serialized_params)
    python_params = {}
    for param in proto_struct.parameters:
        python_value: Any
        if param.type == api_pb2.PARAM_TYPE_STRING:
            python_value = param.string_value
        elif param.type == api_pb2.PARAM_TYPE_INT:
            python_value = param.int_value
        elif param.type == api_pb2.PARAM_TYPE_BYTES:
            python_value = param.bytes_value
        else:
            raise NotImplementedError(f"Unimplemented parameter type: {param.type}.")

        python_params[param.name] = python_value

    return python_params


def validate_parameter_values(payload: dict[str, Any], schema: typing.Sequence[api_pb2.ClassParameterSpec]):
    """Ensure parameter payload conforms to the schema of a class

    Checks that:
    * All fields are specified (defaults are expected to already be applied on the payload)
    * No extra fields are specified
    * The type of each field is correct
    """
    for schema_param in schema:
        if schema_param.name not in payload:
            raise InvalidError(f"Missing required parameter: {schema_param.name}")
        python_value = payload[schema_param.name]
        decoder = type_register.get_decoder(schema_param.type)  # use the schema's expected decoder
        encoder = type_register.get_encoder(python_value)
        if decoder != encoder:
            got_type = type(python_value)
            compatible_types = type_register._python_types_for_handler(decoder)
            compatible_types_str = "|".join(t.__name__ for t in compatible_types)
            raise TypeError(
                f"Parameter '{schema_param.name}' type error: Expected {compatible_types_str}, got {got_type.__name__}"
            )

    schema_fields = {p.name for p in schema}
    # then check that no extra values are provided
    non_declared_fields = payload.keys() - schema_fields
    if non_declared_fields:
        raise InvalidError(
            f"The following parameter names were provided but are not present in the schema: {non_declared_fields}"
        )


def deserialize_params(serialized_params: bytes, function_def: api_pb2.Function, _client: "modal.client._Client"):
    if function_def.class_parameter_info.format in (
        api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_UNSPECIFIED,
        api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PICKLE,
    ):
        # legacy serialization format - pickle of `(args, kwargs)` w/ support for modal object arguments
        param_args, param_kwargs = deserialize(serialized_params, _client)
    elif function_def.class_parameter_info.format == api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PROTO:
        param_args = ()  # we use kwargs only for our implicit constructors
        param_kwargs = deserialize_proto_params(serialized_params)
    else:
        raise ExecutionError(
            f"Unknown class parameter serialization format: {function_def.class_parameter_info.format}"
        )

    return param_args, param_kwargs


PH = typing.TypeVar("PH", bound=type["PayloadHandler"])


class TypeRegistry:
    _py_base_type_to_handler: dict[type, "PayloadHandler"]
    _proto_type_to_handler: dict["api_pb2.ParameterType.ValueType", "PayloadHandler"]

    def __init__(self):
        self._py_base_type_to_handler = {}
        self._proto_type_to_handler = {}

    def register_encoder(self, python_base_type: type) -> typing.Callable[[PH], PH]:
        def deco(ph: type[PayloadHandler]):
            if python_base_type in self._py_base_type_to_handler:
                raise ValueError("Can't register the same encoder type twice")
            self._py_base_type_to_handler[python_base_type] = ph.singleton()
            return ph

        return deco

    def register_decoder(self, enum_type_value: "api_pb2.ParameterType.ValueType"):
        def deco(ph: type[PayloadHandler]):
            if enum_type_value in self._proto_type_to_handler:
                raise ValueError("Can't register the same decoder type twice")
            self._proto_type_to_handler[enum_type_value] = ph.singleton()
            return ph

        return deco

    def _get_for_base_type(self, python_base_type: type) -> "PayloadHandler":
        # private helper method
        if handler := self._py_base_type_to_handler.get(python_base_type):
            return handler

        return UnknownTypeHandler.singleton()

    def get_for_declared_type(self, python_declared_type: type) -> "PayloadHandler":
        """PayloadHandler for a type annotation (that could be a generic type)"""
        if origin := typing_extensions.get_origin(python_declared_type):
            base_type = origin  # look up by generic type, not exact type
        else:
            base_type = python_declared_type

        return self._get_for_base_type(base_type)

    def get_encoder(self, python_runtime_value: Any) -> "PayloadHandler":
        return self._get_for_base_type(type(python_runtime_value))

    def get_decoder(self, proto_type_enum: "api_pb2.ParameterType.ValueType") -> "PayloadHandler":
        if handler := self._proto_type_to_handler.get(proto_type_enum):
            return handler

        if proto_type_enum not in api_pb2.ParameterType.values():
            raise InvalidError(
                f"modal {client_version} doesn't recognize payload type {proto_type_enum}. "
                f"Try upgrading to a later version of Modal."
            )

        enum_name = api_pb2.ParameterType.Name(proto_type_enum)
        raise InvalidError(f"No payload handler implemented for payload type {enum_name}")

    def _python_types_for_handler(self, payload_handler: "PayloadHandler") -> set[type]:
        # reverse lookup of which types map to a specific handler
        return {k for k, h in self._py_base_type_to_handler.items() if h == payload_handler}


type_register = TypeRegistry()


class PayloadHandler(metaclass=abc.ABCMeta):
    allow_as_class_parameter = False
    _singleton: typing.ClassVar[typing.Optional[typing_extensions.Self]] = None

    @abstractmethod
    def encode(self, python_value: Any) -> api_pb2.ClassParameterValue: ...

    @abstractmethod
    def decode(self, struct: api_pb2.ClassParameterValue) -> Any: ...

    @abstractmethod
    def proto_type_def(self, declared_python_type: type) -> api_pb2.GenericPayloadType: ...

    @classmethod
    def singleton(cls) -> typing_extensions.Self:
        # a bit hacky, but this lets us register the same instance multiple times using decorators
        if "_singleton" not in cls.__dict__:  # use .__dict__ to avoid parent class resolution of getattr
            cls._singleton = cls()
        return cls._singleton


@type_register.register_encoder(int)
@type_register.register_decoder(api_pb2.PARAM_TYPE_INT)
class IntType(PayloadHandler):
    allow_as_class_parameter = True

    def encode(self, i: int) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_INT, int_value=i)

    def decode(self, p: api_pb2.ClassParameterValue) -> int:
        return p.int_value

    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        assert full_python_type is int
        return api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_INT,
        )


@type_register.register_encoder(str)
@type_register.register_decoder(api_pb2.PARAM_TYPE_STRING)
class StringType(PayloadHandler):
    allow_as_class_parameter = True

    def encode(self, s: str) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_STRING, string_value=s)

    def decode(self, p: api_pb2.ClassParameterValue) -> str:
        return p.string_value

    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        assert full_python_type is str
        return api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_STRING,
        )


@type_register.register_encoder(bytes)
@type_register.register_decoder(api_pb2.PARAM_TYPE_BYTES)
class BytesType(PayloadHandler):
    allow_as_class_parameter = True

    def encode(self, b: bytes) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_BYTES, bytes_value=b)

    def decode(self, p: api_pb2.ClassParameterValue) -> bytes:
        return p.bytes_value

    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        assert full_python_type is bytes
        return api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_BYTES,
        )


class UnknownTypeHandler(PayloadHandler):
    # we could potentially use this to encode/decode values as pickle bytes
    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        return api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_UNKNOWN)

    def encode(self, python_value: Any) -> api_pb2.ClassParameterValue:
        # TODO: we could use pickle here?
        raise NotImplementedError(f"Can't encode unknown type {type(python_value)}")

    def decode(self, struct: api_pb2.ClassParameterValue) -> Any:
        raise NotImplementedError(f"Can't decode unknown value {struct.type}")


@type_register.register_encoder(list)  # might want to do tuple separately
@type_register.register_decoder(api_pb2.PARAM_TYPE_LIST)
class ListType(PayloadHandler):
    def encode(self, value: list) -> api_pb2.ClassParameterValue:
        item_values = []
        for item in value:
            item_values.append(type_register.get_encoder(item).encode(item))
        return api_pb2.ClassParameterValue(
            type=api_pb2.PARAM_TYPE_LIST, list_value=api_pb2.PayloadListValue(items=item_values)
        )

    def _decode_list_value(self, list_value: api_pb2.PayloadListValue) -> list:
        item_values = []
        for item in list_value.items:
            item_values.append(type_register.get_decoder(item.type).decode(item))
        return item_values

    def decode(self, p: api_pb2.ClassParameterValue) -> list:
        return self._decode_list_value(p.list_value)

    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        origin = typing_extensions.get_origin(full_python_type)  # expected to be `list`
        args = typing_extensions.get_args(full_python_type)
        if origin and len(args) == 1:
            sub_type_handler = type_register.get_for_declared_type(args[0])
            arg = args[0]
        else:
            sub_type_handler = UnknownTypeHandler()
            arg = typing.Any

        return api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_LIST, sub_types=[sub_type_handler.proto_type_def(arg)]
        )


@type_register.register_encoder(dict)
@type_register.register_decoder(api_pb2.PARAM_TYPE_DICT)
class DictType(PayloadHandler):
    def encode(self, value: dict[str, Any]) -> api_pb2.ClassParameterValue:
        item_values = []
        for key, item in value.items():
            item = type_register.get_encoder(item).encode(item)
            item.name = key
            item_values.append(item)

        return api_pb2.ClassParameterValue(
            type=api_pb2.PARAM_TYPE_DICT,
            dict_value=api_pb2.PayloadDictValue(
                entries=item_values,
            ),
        )

    def _decode_dict_value(self, dict_value: api_pb2.PayloadDictValue) -> dict[str, Any]:
        item_values = {}
        for item in dict_value.entries:
            item_values[item.name] = type_register.get_decoder(item.type).decode(item)
        return item_values

    def decode(self, p: api_pb2.ClassParameterValue) -> dict[str, Any]:
        return self._decode_dict_value(p.dict_value)

    def proto_type_def(self, full_python_type: type) -> api_pb2.GenericPayloadType:
        origin = typing_extensions.get_origin(full_python_type)  # expected to be `dict`
        args = typing_extensions.get_args(full_python_type)
        if origin and len(args) == 2:
            if args[0] is not str:
                raise InvalidError("Dict arguments only allow str keys")
            sub_type_handler = type_register.get_for_declared_type(args[1])
            arg = args[1]
        else:
            sub_type_handler = UnknownTypeHandler()
            arg = typing.Any

        return api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_DICT, sub_types=[sub_type_handler.proto_type_def(arg)]
        )


def _signature_parameter_to_spec(
    python_signature_parameter: inspect.Parameter, include_legacy_parameter_fields: bool = False
) -> api_pb2.ClassParameterSpec:
    python_type = python_signature_parameter.annotation

    origin = typing_extensions.get_origin(python_type)
    if origin:
        base_type = origin
    else:
        base_type = python_type

    payload_handler = type_register.get_for_declared_type(base_type)

    full_proto_type = payload_handler.proto_type_def(python_type)
    has_default = python_signature_parameter.default is not Parameter.empty

    if has_default:
        maybe_default_value = type_register.get_encoder(python_signature_parameter.default).encode(
            python_signature_parameter.default
        )
    else:
        maybe_default_value = None

    field_spec = api_pb2.ClassParameterSpec(
        name=python_signature_parameter.name,
        full_type=full_proto_type,
        has_default=has_default,
        default_value=maybe_default_value,
    )
    if include_legacy_parameter_fields:
        # For backward compatibility reasons with clients that don't look at .default_value:
        # We need to still provide defaults for int, str and bytes in the base object
        # We can remove this when all supported clients + backend only look at .default_value and .full_type

        if full_proto_type.base_type == api_pb2.PARAM_TYPE_INT:
            if has_default:
                field_spec.int_default = python_signature_parameter.default
            field_spec.type = api_pb2.PARAM_TYPE_INT
        elif full_proto_type.base_type == api_pb2.PARAM_TYPE_STRING:
            if has_default:
                field_spec.string_default = python_signature_parameter.default
            field_spec.type = api_pb2.PARAM_TYPE_STRING
        elif full_proto_type.base_type == api_pb2.PARAM_TYPE_BYTES:
            if has_default:
                field_spec.bytes_default = python_signature_parameter.default
            field_spec.type = api_pb2.PARAM_TYPE_BYTES

    return field_spec


def signature_to_parameter_specs(signature: inspect.Signature) -> list[api_pb2.ClassParameterSpec]:
    # only used for modal.parameter() specs, uses backwards compatible
    modal_parameters: list[api_pb2.ClassParameterSpec] = []
    for param in signature.parameters.values():
        field_spec = _signature_parameter_to_spec(param, include_legacy_parameter_fields=True)
        modal_parameters.append(field_spec)
    return modal_parameters


def validate_parameter_type(declared_type: type):
    """Raises a helpful TypeError if the supplied type isn't supported by class parameters"""
    supported_types = [
        base_type
        for base_type, handler in type_register._py_base_type_to_handler.items()
        if handler.allow_as_class_parameter
    ]
    supported_str = ", ".join(t.__name__ for t in supported_types)

    if (
        not (payload_handler := type_register.get_for_declared_type(declared_type))
        or not payload_handler.allow_as_class_parameter
    ):
        raise TypeError(
            f"{declared_type.__name__} is not a supported modal.parameter() type. Use one of: {supported_str}"
        )
