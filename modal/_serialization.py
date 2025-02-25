# Copyright Modal Labs 2022
import io
import pickle
import typing
from dataclasses import dataclass
from typing import Any

from modal._utils.async_utils import synchronizer
from modal_proto import api_pb2

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
    elif data_format == api_pb2.DATA_FORMAT_ASGI:
        return _serialize_asgi(obj).SerializeToString(deterministic=True)
    elif data_format == api_pb2.DATA_FORMAT_GENERATOR_DONE:
        assert isinstance(obj, api_pb2.GeneratorDone)
        return obj.SerializeToString(deterministic=True)
    else:
        raise InvalidError(f"Unknown data format {data_format!r}")


def deserialize_data_format(s: bytes, data_format: int, client) -> Any:
    if data_format == api_pb2.DATA_FORMAT_PICKLE:
        return deserialize(s, client)
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


@dataclass
class ParamTypeInfo:
    default_field: str
    proto_field: str
    encoder: typing.Callable[[Any], Any]
    decoder: typing.Callable[[Any, "modal.client._Client"], Any]


PYTHON_TO_PROTO_TYPE: dict[type, "api_pb2.ParameterType.ValueType"] = {
    str: api_pb2.PARAM_TYPE_STRING,
    int: api_pb2.PARAM_TYPE_INT,
    list: api_pb2.PARAM_TYPE_LIST,
    dict: api_pb2.PARAM_TYPE_DICT,
}


class PayloadHandler:
    _handlers = []  # list of (type, ParameterType, PayloadHandler)
    default_field: str

    def __init_subclass__(cls, t, e):
        PayloadHandler._handlers.append((t, e, cls()))

    def serialize(self, python_value: Any) -> api_pb2.ClassParameterValue:
        for t, e, c in PayloadHandler._handlers:
            if isinstance(python_value, t):
                return c.serialize(python_value)
        else:
            raise RuntimeError(f"can't serialize type {type(python_value)}")

    def deserialize(self, pv: api_pb2.ClassParameterValue, client) -> Any:
        for t, e, c in PayloadHandler._handlers:
            if pv.type == e:
                return c.deserialize(pv, client)
        else:
            raise RuntimeError(f"can't deserialize type {pv.type}")

    def for_proto_type(self, proto_type: "api_pb2.ParameterType.ValueType") -> "PayloadHandler":
        for t, e, c in PayloadHandler._handlers:
            if proto_type == e:
                return c
        raise ValueError(f"No support for {proto_type}")


payload_handler = PayloadHandler()


class StringPayloadHandler(PayloadHandler, t=str, e=api_pb2.PARAM_TYPE_STRING):
    default_field = "string_default"

    def serialize(self, python_value: str) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_STRING, string_value=python_value)

    def deserialize(self, value: api_pb2.ClassParameterValue, client) -> str:
        return value.str_value


class IntPayloadHandler(PayloadHandler, t=int, e=api_pb2.PARAM_TYPE_INT):
    default_field = "int_default"

    def serialize(self, python_value: int) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_INT, int_value=python_value)

    def deserialize(self, value: api_pb2.ClassParameterValue, client) -> int:
        return value.int_value


class BoolPayloadHandler(PayloadHandler, t=bool, e=api_pb2.PARAM_TYPE_BOOL):
    def serialize(self, python_value: bool) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_BOOL, bool_value=python_value)

    def deserialize(self, value: api_pb2.ClassParameterValue, client) -> bool:
        return value.bool_value


class FloatPayloadHandler(PayloadHandler, t=float, e=api_pb2.PARAM_TYPE_FLOAT):
    def serialize(self, python_value: float) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_FLOAT, float_value=python_value)

    def deserialize(self, value: api_pb2.ClassParameterValue, client) -> float:
        return value.float_value


class BytesPayloadHandler(PayloadHandler, t=bytes, e=api_pb2.PARAM_TYPE_BYTES):
    def serialize(self, python_value: bytes) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_BYTES, bytes_value=python_value)

    def deserialize(self, value: api_pb2.ClassParameterValue, client) -> bytes:
        return value.bytes_value


NoneType = type(None)


class NonePayloadHandler(PayloadHandler, t=NoneType, e=api_pb2.PARAM_TYPE_NONE):
    def serialize(self, python_value: NoneType) -> api_pb2.ClassParameterValue:
        return api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_NONE)

    def deserialize(self, value: api_pb2.ClassParameterValue, client) -> NoneType:
        return None


class ListPayloadHandler(PayloadHandler, t=list, e=api_pb2.PARAM_TYPE_LIST):
    def serialize(self, python_value: list) -> api_pb2.ClassParameterValue:
        values: list[api_pb2.ClassParameterValue] = [payload_handler.serialize(v) for v in python_value]
        return api_pb2.ClassParameterValue(
            type=api_pb2.PARAM_TYPE_LIST, list_value=api_pb2.PayloadListValue(values=values)
        )

    def deserialize(self, value: api_pb2.ClassParameterValue, client) -> list:
        return [payload_handler.deserialize(item, client) for item in value.list_value.items]


class DictPayloadHandler(PayloadHandler, t=dict, e=api_pb2.PARAM_TYPE_DICT):
    def serialize(self, python_value: dict[str, Any]) -> api_pb2.ClassParameterValue:
        entries: list[api_pb2.PayloadDictEntry] = [
            api_pb2.PayloadDictEntry(name=k, value=payload_handler.serialize(v)) for k, v in python_value
        ]
        return api_pb2.ClassParameterValue(
            type=api_pb2.PARAM_TYPE_DICT, dict_value=api_pb2.PayloadDictValue(entries=entries)
        )

    def deserialize(self, value: api_pb2.ClassParameterValue, client) -> dict:
        values: list = [payload_handler.deserialize(value, client) for value in value.dict_value.values]
        return dict(zip(value.dict_value.keys, values))


PROTO_TYPE_INFO = {}


def serialize_proto_params(python_params: dict[str, Any], schema: typing.Sequence[api_pb2.ClassParameterSpec]) -> bytes:
    proto_params: list[api_pb2.ClassParameterValue] = []
    for schema_param in schema:
        handler = payload_handler.for_proto_type(schema_param.type)
        python_value = python_params.get(schema_param.name)
        if python_value is None:
            if schema_param.has_default:
                # NOTE: this python_value is actually a proto value that's not packaged in a ClassParameterValue
                #   which happens to work for int/str which are the only supported types at the moment
                #   TODO: we should fix this, maybe having a single ClassParameterValue default_value?
                python_value = getattr(schema_param, handler.default_field)
            else:
                raise ValueError(f"Missing required parameter: {schema_param.name}")
        proto_param = handler.serialize(python_value)
        proto_param.name = schema_param.name
        proto_params.append(proto_param)

    proto_bytes = api_pb2.ClassParameterSet(parameters=proto_params).SerializeToString(deterministic=True)
    return proto_bytes


def python_to_proto_payload(python_args: tuple[Any, ...], python_kwargs: dict[str, Any]) -> api_pb2.Payload:
    """Serialize a python payload using the input payload type rather than a schema w/ type coercion/validation

    This is similar to serialize_proto_params except:
    * Doesn't require a prior schema for encoding
    * It uses the new api_pb2.Payload container proto to include both args and kwargs
    * Doesn't use the `name` field of the ClassParameterValue message (names are encoded as part
      of the `kwargs` PayloadDictValue instead)
    """
    proto_args = payload_handler.serialize(list(python_args))
    proto_kwargs = payload_handler.serialize(python_kwargs)
    return api_pb2.Payload(
        args=proto_args.list_value,
        kwargs=proto_kwargs.dict_value,
    )


def proto_to_python_payload(
    proto_payload: api_pb2.Payload, client: "modal.client._Client"
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    python_args = []
    for proto_value in proto_payload.args.values:
        python_value = _proto_to_python_value(proto_value, client)
        python_args.append(python_value)

    python_kwargs = {}
    for proto_dict_entry in proto_payload.kwargs.entries:
        python_value = _proto_to_python_value(proto_dict_entry.value, client)
        python_kwargs[proto_dict_entry.name] = python_value

    return tuple(python_args), python_kwargs


def deserialize_proto_params(serialized_params: bytes, schema: list[api_pb2.ClassParameterSpec]) -> dict[str, Any]:
    # TODO: this currently requires the schema to decode a payload, but we could separate validation from decoding
    proto_struct = api_pb2.ClassParameterSet()
    proto_struct.ParseFromString(serialized_params)
    value_by_name = {p.name: p for p in proto_struct.parameters}
    python_params = {}
    for schema_param in schema:
        if schema_param.name not in value_by_name:
            # TODO: handle default values? Could just be a flag on the FunctionParameter schema spec,
            #  allowing it to not be supplied in the FunctionParameterSet?
            raise AttributeError(f"Constructor arguments don't match declared parameters (missing {schema_param.name})")
        param_value = value_by_name[schema_param.name]
        if schema_param.type != param_value.type:
            raise ValueError(
                "Constructor arguments types don't match declared parameters "
                f"({schema_param.name}: type {schema_param.type} != type {param_value.type})"
            )
        python_value: Any
        if schema_param.type == api_pb2.PARAM_TYPE_STRING:
            python_value = param_value.string_value
        elif schema_param.type == api_pb2.PARAM_TYPE_INT:
            python_value = param_value.int_value
        else:
            # TODO(elias): based on `parameters` declared types, we could add support for
            #  custom non proto types encoded as bytes in the proto, e.g. PARAM_TYPE_PYTHON_PICKLE
            raise NotImplementedError("Only strings and ints are supported parameter value types at the moment")

        python_params[schema_param.name] = python_value

    return python_params


def deserialize_params(serialized_params: bytes, function_def: api_pb2.Function, _client: "modal.client._Client"):
    if function_def.class_parameter_info.format in (
        api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_UNSPECIFIED,
        api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PICKLE,
    ):
        # legacy serialization format - pickle of `(args, kwargs)` w/ support for modal object arguments
        param_args, param_kwargs = deserialize(serialized_params, _client)
    elif function_def.class_parameter_info.format == api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PROTO:
        param_args = ()
        param_kwargs = deserialize_proto_params(serialized_params, list(function_def.class_parameter_info.schema))
    else:
        raise ExecutionError(
            f"Unknown class parameter serialization format: {function_def.class_parameter_info.format}"
        )

    return param_args, param_kwargs
