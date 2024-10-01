# Copyright Modal Labs 2024
# TODO explain!!!
import google.protobuf
pb_version = int(google.protobuf.__version__[0])

if pb_version == 3:
    from .pb3.modal_proto.api_pb2_grpc import *
elif pb_version == 4:
    from .pb4.modal_proto.api_pb2_grpc import *
elif pb_version == 5:
    from .pb5.modal_proto.api_pb2_grpc import *
else:
    ...  # TODO what should happen here?