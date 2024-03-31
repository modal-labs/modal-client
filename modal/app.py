# Copyright Modal Labs 2022
from typing import TYPE_CHECKING, Dict, List, Optional, TypeVar

from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import Message

from modal_proto import api_pb2

from ._output import OutputManager
from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import get_proto_oneof, retry_transient_errors
from .client import _Client
from .config import logger
from .exception import ExecutionError, InvalidError
from .object import _Object

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

    async def _create_all_objects(
        self,
        indexed_objects: Dict[str, _Object],
        new_app_state: int,
        environment_name: str,
        output_mgr: Optional[OutputManager] = None,
    ):  # api_pb2.AppState.V
        """Create objects that have been defined but not created on the server."""
        if not self.client.authenticated:
            raise ExecutionError("Objects cannot be created with an unauthenticated client")

        resolver = Resolver(
            self.client,
            output_mgr=output_mgr,
            environment_name=environment_name,
            app_id=self.app_id,
        )
        with resolver.display():
            # Get current objects, and reset all objects
            tag_to_object_id = self.tag_to_object_id
            self.tag_to_object_id = {}

            # Assign all objects
            for tag, obj in indexed_objects.items():
                # Reset object_id in case the app runs twice
                # TODO(erikbern): clean up the interface
                obj._unhydrate()

            # Preload all functions to make sure they have ids assigned before they are loaded.
            # This is important to make sure any enclosed function handle references in serialized
            # functions have ids assigned to them when the function is serialized.
            # Note: when handles/objs are merged, all objects will need to get ids pre-assigned
            # like this in order to be referrable within serialized functions
            for tag, obj in indexed_objects.items():
                existing_object_id = tag_to_object_id.get(tag)
                # Note: preload only currently implemented for Functions, returns None otherwise
                # this is to ensure that directly referenced functions from the global scope has
                # ids associated with them when they are serialized into other functions
                await resolver.preload(obj, existing_object_id)
                if obj.object_id is not None:
                    tag_to_object_id[tag] = obj.object_id

            for tag, obj in indexed_objects.items():
                existing_object_id = tag_to_object_id.get(tag)
                await resolver.load(obj, existing_object_id)
                self.tag_to_object_id[tag] = obj.object_id

        # Create the app (and send a list of all tagged obs)
        # TODO(erikbern): we should delete objects from a previous version that are no longer needed
        # We just delete them from the app, but the actual objects will stay around
        indexed_object_ids = self.tag_to_object_id
        assert indexed_object_ids == self.tag_to_object_id
        all_objects = resolver.objects()

        unindexed_object_ids = list(set(obj.object_id for obj in all_objects) - set(self.tag_to_object_id.values()))
        req_set = api_pb2.AppSetObjectsRequest(
            app_id=self.app_id,
            indexed_object_ids=indexed_object_ids,
            unindexed_object_ids=unindexed_object_ids,
            new_app_state=new_app_state,  # type: ignore
        )
        await retry_transient_errors(self.client.stub.AppSetObjects, req_set)

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

    def associate_stub_container(self, stub):
        stub._container_app = self

        # Initialize objects on stub
        stub_objects: dict[str, _Object] = dict(stub.get_objects())
        for tag, object_id in self.tag_to_object_id.items():
            obj = stub_objects.get(tag)
            if obj is not None:
                handle_metadata = self.object_handle_metadata[object_id]
                obj._hydrate(object_id, self.client, handle_metadata)

    def _has_object(self, tag: str) -> bool:
        return tag in self.tag_to_object_id

    def _hydrate_object(self, obj, tag: str):
        object_id: str = self.tag_to_object_id[tag]
        metadata: Message = self.object_handle_metadata[object_id]
        obj._hydrate(object_id, self.client, metadata)

    def hydrate_function_deps(self, function: _Function, dep_object_ids: List[str]):
        function_deps = function.deps(only_explicit_mounts=True)
        if len(function_deps) != len(dep_object_ids):
            raise ExecutionError(
                f"Function has {len(function_deps)} dependencies"
                f" but container got {len(dep_object_ids)} object ids."
            )
        for object_id, obj in zip(dep_object_ids, function_deps):
            metadata: Message = self.object_handle_metadata[object_id]
            obj._hydrate(object_id, self.client, metadata)

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
    if _container_app._is_interactivity_enabled:
        # Currently, interactivity is enabled forever
        return
    _container_app._is_interactivity_enabled = True

    if not client:
        client = await _Client.from_env()

    if client.client_type != api_pb2.CLIENT_TYPE_CONTAINER:
        raise InvalidError("Interactivity only works inside a Modal Container.")

    if _container_app._function_def is not None:
        if not _container_app._function_def.pty_info:
            raise InvalidError(
                "Interactivity is not enabled in this function. Use MODAL_INTERACTIVE_FUNCTIONS=1 to enable interactivity."
            )

        if _container_app._function_def.concurrency_limit > 1:
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
