# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modal_proto/options.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import descriptor_pb2 as google_dot_protobuf_dot_descriptor__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19modal_proto/options.proto\x12\rmodal.options\x1a google/protobuf/descriptor.proto:=\n\x11\x61udit_target_attr\x12\x1d.google.protobuf.FieldOptions\x18\xd0\x86\x03 \x01(\x08\x88\x01\x01:=\n\x10\x61udit_event_name\x12\x1e.google.protobuf.MethodOptions\x18\xd0\x86\x03 \x01(\t\x88\x01\x01:D\n\x17\x61udit_event_description\x12\x1e.google.protobuf.MethodOptions\x18\xd1\x86\x03 \x01(\t\x88\x01\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'modal_proto.options_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(audit_target_attr)
  google_dot_protobuf_dot_descriptor__pb2.MethodOptions.RegisterExtension(audit_event_name)
  google_dot_protobuf_dot_descriptor__pb2.MethodOptions.RegisterExtension(audit_event_description)

  DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
