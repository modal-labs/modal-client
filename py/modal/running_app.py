# Copyright Modal Labs 2024
from dataclasses import dataclass, field

from google.protobuf.message import Message

from modal._utils.grpc_utils import get_proto_oneof
from modal_proto import api_pb2


@dataclass
class RunningApp:
    app_id: str
    app_page_url: str | None = None
    app_logs_url: str | None = None
    function_ids: dict[str, str] = field(default_factory=dict)
    class_ids: dict[str, str] = field(default_factory=dict)
    object_handle_metadata: dict[str, Message | None] = field(default_factory=dict)
    interactive: bool = False


def running_app_from_layout(
    app_id: str,
    app_layout: api_pb2.AppLayout,
    app_page_url: str | None = None,
) -> RunningApp:
    object_handle_metadata = {}
    for obj in app_layout.objects:
        handle_metadata: Message | None = get_proto_oneof(obj, "handle_metadata_oneof")
        object_handle_metadata[obj.object_id] = handle_metadata

    return RunningApp(
        app_id,
        function_ids=dict(app_layout.function_ids),
        class_ids=dict(app_layout.class_ids),
        object_handle_metadata=object_handle_metadata,
        app_page_url=app_page_url,
    )
