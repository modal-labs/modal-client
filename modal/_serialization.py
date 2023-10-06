# Copyright Modal Labs 2022
import io
import pickle
from typing import Any

import cloudpickle

from modal_proto import api_pb2

from .config import logger
from .exception import InvalidError
from .object import Object, _Object

PICKLE_PROTOCOL = 4  # Support older Python versions.


class Pickler(cloudpickle.Pickler):
    def __init__(self, buf):
        super().__init__(buf, protocol=PICKLE_PROTOCOL)

    def persistent_id(self, obj):
        if isinstance(obj, _Object):
            flag = "_o"
        elif isinstance(obj, Object):
            flag = "o"
        else:
            return
        if not obj.object_id:
            raise InvalidError(f"Can't serialize object {obj} which hasn't been created.")
        return (obj.object_id, flag, obj._get_metadata())


class Unpickler(pickle.Unpickler):
    def __init__(self, client, buf):
        self.client = client
        super().__init__(buf)

    def persistent_load(self, pid):
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
    return Unpickler(client, io.BytesIO(s)).load()


def _serialize_asgi(obj: Any) -> api_pb2.Asgi:
    def flatten_headers(obj):
        return [s for k, v in obj for s in (k, v)]

    if obj is None:
        return api_pb2.Asgi()

    msg_type = obj.get("type")
    if msg_type == "http":
        return api_pb2.Asgi(
            http=api_pb2.Asgi.Http(
                http_version=obj["http_version"],
                method=obj["method"],
                scheme=obj.get("scheme"),
                path=obj["path"],
                query_string=obj["query_string"],
                headers=flatten_headers(obj["headers"]),
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
                headers=flatten_headers(obj["headers"]),
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
                headers=flatten_headers(obj["headers"]),
                more_trailers=obj.get("more_trailers"),
            )
        )
    elif msg_type == "http.disconnect":
        return api_pb2.Asgi(http_disconnect=api_pb2.Asgi.HttpDisconnect())
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
    else:
        assert msg_type is None
        return None


def serialize_data_format(obj: Any, data_format: int) -> bytes:
    """Similar to serialize(), but supports other data formats."""
    if data_format == api_pb2.DATA_FORMAT_PICKLE:
        return serialize(obj)
    elif data_format == api_pb2.DATA_FORMAT_ASGI:
        return _serialize_asgi(obj).SerializeToString(deterministic=True)
    else:
        raise InvalidError(f"Unknown data format {data_format!r}")


def deserialize_data_format(s: bytes, data_format: int, client) -> Any:
    if data_format == api_pb2.DATA_FORMAT_UNSPECIFIED:
        # TODO: Remove this after Modal client version 0.52, when the data_format field is always set.
        return deserialize(s, client)
    if data_format == api_pb2.DATA_FORMAT_PICKLE:
        return deserialize(s, client)
    elif data_format == api_pb2.DATA_FORMAT_ASGI:
        asgi = api_pb2.Asgi()
        asgi.ParseFromString(s)
        return _deserialize_asgi(asgi)
    else:
        raise InvalidError(f"Unknown data format {data_format!r}")
