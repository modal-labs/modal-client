# Copyright Modal Labs 2024
from dataclasses import dataclass, field
from typing import Optional

from google.protobuf.message import Message

from modal._utils.grpc_utils import get_proto_oneof
from modal_proto import api_pb2

from .client import _Client


@dataclass
class RunningApp:
    app_id: str
    client: _Client
    environment_name: Optional[str] = None
    app_page_url: Optional[str] = None
    app_logs_url: Optional[str] = None
    tag_to_object_id: dict[str, str] = field(default_factory=dict)
    object_handle_metadata: dict[str, Optional[Message]] = field(default_factory=dict)
    interactive: bool = False


def running_app_from_layout(
    app_id: str,
    app_layout: api_pb2.AppLayout,
    client: _Client,
    environment_name: Optional[str] = None,
    app_page_url: Optional[str] = None,
) -> RunningApp:
    tag_to_object_id = dict(**app_layout.function_ids, **app_layout.class_ids)
    object_handle_metadata = {}
    for obj in app_layout.objects:
        handle_metadata: Optional[Message] = get_proto_oneof(obj, "handle_metadata_oneof")
        object_handle_metadata[obj.object_id] = handle_metadata

    return RunningApp(
        app_id,
        client,
        environment_name=environment_name,
        tag_to_object_id=tag_to_object_id,
        object_handle_metadata=object_handle_metadata,
        app_page_url=app_page_url,
    )
