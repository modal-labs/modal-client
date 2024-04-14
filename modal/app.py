# Copyright Modal Labs 2022
from typing import TYPE_CHECKING, Dict, List, Optional, TypeVar

from google.protobuf.message import Message

from modal_proto import api_pb2

from ._utils.grpc_utils import get_proto_oneof
from .app_utils import _list_apps, list_apps  # noqa: F401
from .config import logger

if TYPE_CHECKING:
    from .functions import _Function

else:
    _Function = TypeVar("_Function")


class _LocalApp:
    tag_to_object_id: Dict[str, str]
    app_id: str
    app_page_url: str
    environment_name: str
    interactive: bool

    def __init__(
        self,
        app_id: str,
        app_page_url: str,
        tag_to_object_id: Optional[Dict[str, str]] = None,
        environment_name: Optional[str] = None,
        interactive: bool = False,
    ):
        """mdmd:hidden This is the app constructor. Users should not call this directly."""
        self.app_id = app_id
        self.app_page_url = app_page_url
        self.tag_to_object_id = tag_to_object_id or {}
        self.environment_name = environment_name
        self.interactive = interactive


class _ContainerApp:
    app_id: Optional[str]
    environment_name: Optional[str]
    tag_to_object_id: Dict[str, str]
    object_handle_metadata: Dict[str, Optional[Message]]
    # if true, there's an active PTY shell session connected to this process.
    is_interactivity_enabled: bool
    function_def: Optional[api_pb2.Function]

    def __init__(self):
        self.app_id = None
        self.environment_name = None
        self.tag_to_object_id = {}
        self.object_handle_metadata = {}
        self.is_interactivity_enabled = False
        self.function_def = None
        self.fetching_inputs = True


def _init_container_app(
    items: List[api_pb2.AppGetObjectsItem],
    app_id: str,
    environment_name: str = "",
    function_def: Optional[api_pb2.Function] = None,
) -> _ContainerApp:
    """Used by the container to bootstrap the app and all its objects. Not intended to be called by Modal users."""
    container_app = _ContainerApp()

    container_app.app_id = app_id
    container_app.environment_name = environment_name
    container_app.function_def = function_def
    container_app.tag_to_object_id = {}
    container_app.object_handle_metadata = {}
    for item in items:
        handle_metadata: Optional[Message] = get_proto_oneof(item.object, "handle_metadata_oneof")
        container_app.object_handle_metadata[item.object.object_id] = handle_metadata
        logger.debug(f"Setting metadata for {item.object.object_id} ({item.tag})")
        if item.tag:
            container_app.tag_to_object_id[item.tag] = item.object.object_id

    return container_app
