# Copyright Modal Labs 2022
from typing import TYPE_CHECKING, Dict, List, Optional, TypeVar

from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import Message

from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import get_proto_oneof, retry_transient_errors
from .client import _Client
from .config import logger
from .exception import InvalidError

if TYPE_CHECKING:
    from .functions import _Function

else:
    _Function = TypeVar("_Function")


class _LocalApp:
    tag_to_object_id: Dict[str, str]
    client: _Client
    app_id: str
    app_page_url: str
    environment_name: str
    interactive: bool

    def __init__(
        self,
        client: _Client,
        app_id: str,
        app_page_url: str,
        tag_to_object_id: Optional[Dict[str, str]] = None,
        environment_name: Optional[str] = None,
        interactive: bool = False,
    ):
        """mdmd:hidden This is the app constructor. Users should not call this directly."""
        self.app_id = app_id
        self.app_page_url = app_page_url
        self.client = client
        self.tag_to_object_id = tag_to_object_id or {}
        self.environment_name = environment_name
        self.interactive = interactive

    async def disconnect(
        self, reason: "Optional[api_pb2.AppDisconnectReason.ValueType]" = None, exc_str: Optional[str] = None
    ):
        """Tell the server the client has disconnected for this app. Terminates all running tasks
        for ephemeral apps."""

        if exc_str:
            exc_str = exc_str[:1000]  # Truncate to 1000 chars

        logger.debug("Sending app disconnect/stop request")
        req_disconnect = api_pb2.AppClientDisconnectRequest(app_id=self.app_id, reason=reason, exception=exc_str)
        await retry_transient_errors(self.client.stub.AppClientDisconnect, req_disconnect)
        logger.debug("App disconnected")


class _ContainerApp:
    client: Optional[_Client]
    app_id: Optional[str]
    environment_name: Optional[str]
    tag_to_object_id: Dict[str, str]
    object_handle_metadata: Dict[str, Optional[Message]]
    # if true, there's an active PTY shell session connected to this process.
    is_interactivity_enabled: bool
    function_def: Optional[api_pb2.Function]
    fetching_inputs: bool

    def __init__(self):
        self.client = None
        self.app_id = None
        self.environment_name = None
        self.tag_to_object_id = {}
        self.object_handle_metadata = {}
        self.is_interactivity_enabled = False
        self.fetching_inputs = True

    async def init(
        self,
        client: _Client,
        app_id: str,
        environment_name: str = "",
        function_def: Optional[api_pb2.Function] = None,
    ):
        """Used by the container to bootstrap the app and all its objects. Not intended to be called by Modal users."""
        global _is_container_app
        _is_container_app = True

        self.client = client
        self.app_id = app_id
        self.environment_name = environment_name
        self.function_def = function_def
        self.tag_to_object_id = {}
        self.object_handle_metadata = {}
        req = api_pb2.AppGetObjectsRequest(app_id=app_id, include_unindexed=True)
        resp = await retry_transient_errors(client.stub.AppGetObjects, req)
        logger.debug(f"AppGetObjects received {len(resp.items)} objects for app {app_id}")
        for item in resp.items:
            handle_metadata: Optional[Message] = get_proto_oneof(item.object, "handle_metadata_oneof")
            self.object_handle_metadata[item.object.object_id] = handle_metadata
            logger.debug(f"Setting metadata for {item.object.object_id} ({item.tag})")
            if item.tag:
                self.tag_to_object_id[item.tag] = item.object.object_id

    @staticmethod
    def _reset_container():
        # Just used for tests
        global _is_container_app, _container_app
        _is_container_app = False
        _container_app.__init__()  # type: ignore

    def stop_fetching_inputs(self):
        self.fetching_inputs = False


LocalApp = synchronize_api(_LocalApp)
ContainerApp = synchronize_api(_ContainerApp)

_is_container_app = False
_container_app = _ContainerApp()
container_app = synchronize_api(_container_app)
assert isinstance(container_app, ContainerApp)


async def _interact(client: Optional[_Client] = None) -> None:
    if _container_app.is_interactivity_enabled:
        # Currently, interactivity is enabled forever
        return
    _container_app.is_interactivity_enabled = True

    if not client:
        client = await _Client.from_env()

    if client.client_type != api_pb2.CLIENT_TYPE_CONTAINER:
        raise InvalidError("Interactivity only works inside a Modal Container.")

    if _container_app.function_def is not None:
        if not _container_app.function_def.pty_info:
            raise InvalidError(
                "Interactivity is not enabled in this function. Use MODAL_INTERACTIVE_FUNCTIONS=1 to enable interactivity."
            )

        if _container_app.function_def.concurrency_limit > 1:
            print(
                "Warning: Interactivity is not supported on functions with concurrency > 1. You may experience unexpected behavior."
            )

    # todo(nathan): add warning if concurrency limit > 1. but idk how to check this here
    # todo(nathan): check if function interactivity is enabled
    try:
        await client.stub.FunctionStartPtyShell(Empty())
    except Exception as e:
        print("Error: Failed to start PTY shell.")
        raise e


interact = synchronize_api(_interact)


def is_local() -> bool:
    """Returns if we are currently on the machine launching/deploying a Modal app

    Returns `True` when executed locally on the user's machine.
    Returns `False` when executed from a Modal container in the cloud.
    """
    return not _is_container_app


async def _list_apps(env: str, client: Optional[_Client] = None) -> List[api_pb2.AppStats]:
    """List apps in a given Modal environment."""
    if client is None:
        client = await _Client.from_env()
    resp: api_pb2.AppListResponse = await client.stub.AppList(api_pb2.AppListRequest(environment_name=env))
    return list(resp.apps)


list_apps = synchronize_api(_list_apps)
