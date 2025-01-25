# Copyright Modal Labs 2024
from typing import Optional

from google.protobuf.message import Message

from modal._utils.grpc_utils import get_proto_oneof
from modal_proto import api_pb2


class RunningApp:
    function_ids: dict[str, str]
    class_ids: dict[str, str]
    object_handle_metadata: dict[str, Optional[Message]]

    def __init__(self, app_layout: Optional[api_pb2.AppLayout] = None):
        self.function_ids = {}
        self.class_ids = {}
        self.object_handle_metadata = {}

        if app_layout:
            for obj in app_layout.objects:
                handle_metadata: Optional[Message] = get_proto_oneof(obj, "handle_metadata_oneof")
                self.object_handle_metadata[obj.object_id] = handle_metadata

            self.function_ids = dict(app_layout.function_ids)
            self.class_ids = dict(app_layout.class_ids)
