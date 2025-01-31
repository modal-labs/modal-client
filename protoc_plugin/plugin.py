#!/usr/bin/env python
# Copyright Modal Labs 2025
# built by modifying grpclib.plugin.main, see https://github.com/vmagamedov/grpclib
# original: Copyright (c) 2019  , Vladimir Magamedov
import os
import sys
from collections import deque
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Deque, NamedTuple, Optional

from google.protobuf.compiler.plugin_pb2 import CodeGeneratorRequest, CodeGeneratorResponse
from google.protobuf.descriptor_pb2 import DescriptorProto, FileDescriptorProto
from grpclib import const

_CARDINALITY = {
    (False, False): const.Cardinality.UNARY_UNARY,
    (True, False): const.Cardinality.STREAM_UNARY,
    (False, True): const.Cardinality.UNARY_STREAM,
    (True, True): const.Cardinality.STREAM_STREAM,
}


class Method(NamedTuple):
    name: str
    cardinality: const.Cardinality
    request_type: str
    reply_type: str


class Service(NamedTuple):
    name: str
    methods: list[Method]


class Buffer:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0

    def add(self, string: str, *args: Any, **kwargs: Any) -> None:
        line = " " * self._indent * 4 + string.format(*args, **kwargs)
        self._lines.append(line.rstrip(" "))

    @contextmanager
    def indent(self) -> Iterator[None]:
        self._indent += 1
        try:
            yield
        finally:
            self._indent -= 1

    def content(self) -> str:
        return "\n".join(self._lines) + "\n"


def render(
    proto_file: str,
    imports: Collection[str],
    services: Collection[Service],
    grpclib_module: str,
) -> str:
    buf = Buffer()
    buf.add("# Generated by the Modal Protocol Buffers compiler. DO NOT EDIT!")
    buf.add("# source: {}", proto_file)
    buf.add("# plugin: {}", __name__)
    if not services:
        return buf.content()

    buf.add("")
    for mod in imports:
        buf.add("import {}", mod)

    buf.add("import typing")
    buf.add("if typing.TYPE_CHECKING:")
    with buf.indent():
        buf.add("import modal.client")

    for service in services:
        buf.add("")
        buf.add("")
        grpclib_stub_name = f"{service.name}Stub"
        buf.add("class {}Modal:", service.name)
        with buf.indent():
            buf.add("")
            buf.add(
                f"def __init__(self, grpclib_stub: {grpclib_module}.{grpclib_stub_name}, "
                + """client: "modal.client._Client") -> None:"""
            )
            with buf.indent():
                if len(service.methods) == 0:
                    buf.add("pass")
                for method in service.methods:
                    name, cardinality, request_type, reply_type = method
                    wrapper_cls: str
                    if cardinality is const.Cardinality.UNARY_UNARY:
                        wrapper_cls = "modal.client.UnaryUnaryWrapper"
                    elif cardinality is const.Cardinality.UNARY_STREAM:
                        wrapper_cls = "modal.client.UnaryStreamWrapper"
                    # elif cardinality is const.Cardinality.STREAM_UNARY:
                    #     wrapper_cls = StreamUnaryWrapper
                    # elif cardinality is const.Cardinality.STREAM_STREAM:
                    #     wrapper_cls = StreamStreamWrapper
                    else:
                        raise TypeError(cardinality)

                    original_method = f"grpclib_stub.{name}"
                    buf.add(f"self.{name} = {wrapper_cls}({original_method}, client)")

    return buf.content()


def _get_proto(request: CodeGeneratorRequest, name: str) -> FileDescriptorProto:
    return next(f for f in request.proto_file if f.name == name)


def _strip_proto(proto_file_path: str) -> str:
    for suffix in [".protodevel", ".proto"]:
        if proto_file_path.endswith(suffix):
            return proto_file_path[: -len(suffix)]

    return proto_file_path


def _base_module_name(proto_file_path: str) -> str:
    basename = _strip_proto(proto_file_path)
    return basename.replace("-", "_").replace("/", ".")


def _proto2pb2_module_name(proto_file_path: str) -> str:
    return _base_module_name(proto_file_path) + "_pb2"


def _proto2grpc_module_name(proto_file_path: str) -> str:
    return _base_module_name(proto_file_path) + "_grpc"


def _type_names(
    proto_file: FileDescriptorProto,
    message_type: DescriptorProto,
    parents: Optional[Deque[str]] = None,
) -> Iterator[tuple[str, str]]:
    if parents is None:
        parents = deque()

    proto_name_parts = [""]
    if proto_file.package:
        proto_name_parts.append(proto_file.package)
    proto_name_parts.extend(parents)
    proto_name_parts.append(message_type.name)

    py_name_parts = [_proto2pb2_module_name(proto_file.name)]
    py_name_parts.extend(parents)
    py_name_parts.append(message_type.name)

    yield ".".join(proto_name_parts), ".".join(py_name_parts)

    parents.append(message_type.name)
    for nested in message_type.nested_type:
        yield from _type_names(proto_file, nested, parents=parents)
    parents.pop()


def main() -> None:
    with os.fdopen(sys.stdin.fileno(), "rb") as inp:
        request = CodeGeneratorRequest.FromString(inp.read())

    types_map: dict[str, str] = {}
    for pf in request.proto_file:
        for mt in pf.message_type:
            types_map.update(_type_names(pf, mt))

    response = CodeGeneratorResponse()

    # See https://github.com/protocolbuffers/protobuf/blob/v3.12.0/docs/implementing_proto3_presence.md  # noqa
    if hasattr(CodeGeneratorResponse, "Feature"):
        response.supported_features = CodeGeneratorResponse.FEATURE_PROTO3_OPTIONAL

    for file_to_generate in request.file_to_generate:
        proto_file = _get_proto(request, file_to_generate)
        module_name = _proto2grpc_module_name(file_to_generate)
        grpclib_module_path = Path(module_name.replace(".", "/") + ".py")

        imports = ["modal._utils.grpc_utils", module_name]

        services = []
        for service in proto_file.service:
            methods = []
            for method in service.method:
                cardinality = _CARDINALITY[(method.client_streaming, method.server_streaming)]
                methods.append(
                    Method(
                        name=method.name,
                        cardinality=cardinality,
                        request_type=types_map[method.input_type],
                        reply_type=types_map[method.output_type],
                    )
                )
            services.append(Service(name=service.name, methods=methods))

        file = response.file.add()

        file.name = str(grpclib_module_path.with_name("modal_" + grpclib_module_path.name))
        file.content = render(
            proto_file=proto_file.name, imports=imports, services=services, grpclib_module=module_name
        )

    with os.fdopen(sys.stdout.fileno(), "wb") as out:
        out.write(response.SerializeToString())


if __name__ == "__main__":
    main()
