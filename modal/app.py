# Copyright Modal Labs 2022
from datetime import date
from typing import TYPE_CHECKING, Any, Dict, Optional, TypeVar

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import get_proto_oneof, retry_transient_errors

from ._output import OutputManager
from ._resolver import Resolver
from .client import _Client
from .config import logger
from .exception import InvalidError, deprecation_error
from .object import _Object

if TYPE_CHECKING:
    from rich.tree import Tree

else:
    Tree = TypeVar("Tree")


class _LocalApp:
    _tag_to_object_id: Dict[str, str]
    _client: _Client
    _app_id: str
    _app_page_url: str
    _environment_name: str

    def __init__(
        self,
        client: _Client,
        app_id: str,
        app_page_url: str,
        tag_to_object_id: Optional[Dict[str, str]] = None,
        stub_name: Optional[str] = None,
        environment_name: Optional[str] = None,
    ):
        """mdmd:hidden This is the app constructor. Users should not call this directly."""
        self._app_id = app_id
        self._app_page_url = app_page_url
        self._client = client
        self._tag_to_object_id = tag_to_object_id or {}
        self._stub_name = stub_name
        self._environment_name = environment_name

    @property
    def client(self) -> _Client:
        """A reference to the running App's server client."""
        return self._client

    @property
    def app_id(self) -> str:
        """A unique identifier for this running App."""
        return self._app_id

    async def _create_all_objects(
        self,
        blueprint: Dict[str, _Object],
        new_app_state: int,
        environment_name: str,
        shell: bool = False,
        output_mgr: Optional[OutputManager] = None,
    ):  # api_pb2.AppState.V
        """Create objects that have been defined but not created on the server."""
        resolver = Resolver(
            self._client,
            output_mgr=output_mgr,
            environment_name=environment_name,
            app_id=self.app_id,
            shell=shell,
        )
        with resolver.display():
            # Get current objects, and reset all objects
            tag_to_object_id = self._tag_to_object_id
            self._tag_to_object_id = {}

            # Assign all objects
            for tag, obj in blueprint.items():
                # Reset object_id in case the app runs twice
                # TODO(erikbern): clean up the interface
                obj._unhydrate()

            # Preload all functions to make sure they have ids assigned before they are loaded.
            # This is important to make sure any enclosed function handle references in serialized
            # functions have ids assigned to them when the function is serialized.
            # Note: when handles/objs are merged, all objects will need to get ids pre-assigned
            # like this in order to be referrable within serialized functions
            for tag, obj in blueprint.items():
                existing_object_id = tag_to_object_id.get(tag)
                # Note: preload only currently implemented for Functions, returns None otherwise
                # this is to ensure that directly referenced functions from the global scope has
                # ids associated with them when they are serialized into other functions
                await resolver.preload(obj, existing_object_id)
                if obj.object_id is not None:
                    tag_to_object_id[tag] = obj.object_id

            for tag, obj in blueprint.items():
                existing_object_id = tag_to_object_id.get(tag)
                await resolver.load(obj, existing_object_id)
                self._tag_to_object_id[tag] = obj.object_id

        # Create the app (and send a list of all tagged obs)
        # TODO(erikbern): we should delete objects from a previous version that are no longer needed
        # We just delete them from the app, but the actual objects will stay around
        indexed_object_ids = self._tag_to_object_id
        assert indexed_object_ids == self._tag_to_object_id
        all_objects = resolver.objects()

        unindexed_object_ids = list(set(obj.object_id for obj in all_objects) - set(self._tag_to_object_id.values()))
        req_set = api_pb2.AppSetObjectsRequest(
            app_id=self._app_id,
            indexed_object_ids=indexed_object_ids,
            unindexed_object_ids=unindexed_object_ids,
            new_app_state=new_app_state,  # type: ignore
        )
        await retry_transient_errors(self._client.stub.AppSetObjects, req_set)

    async def disconnect(self):
        """Tell the server the client has disconnected for this app. Terminates all running tasks
        for ephemeral apps."""

        logger.debug("Sending app disconnect/stop request")
        req_disconnect = api_pb2.AppClientDisconnectRequest(app_id=self._app_id)
        await retry_transient_errors(self._client.stub.AppClientDisconnect, req_disconnect)
        logger.debug("App disconnected")

    async def stop(self):
        """Tell the server to stop this app, terminating all running tasks."""
        req_disconnect = api_pb2.AppStopRequest(app_id=self._app_id)
        await retry_transient_errors(self._client.stub.AppStop, req_disconnect)

    def log_url(self):
        """URL link to a running app's logs page in the Modal dashboard."""
        return self._app_page_url

    def __getitem__(self, tag: str) -> _Object:
        deprecation_error(date(2023, 8, 10), "`app[...]` is no longer supported. Use the stub to get objects instead.")

    def __contains__(self, tag: str) -> bool:
        deprecation_error(
            date(2023, 8, 10), "`obj in app` is no longer supported. Use the stub to get objects instead."
        )

    def __getattr__(self, tag: str) -> _Object:
        if tag.startswith("__"):
            raise AttributeError(f"No such attribute `{tag}`")  # Dumb workaround for doc thing
        deprecation_error(date(2023, 8, 10), "`app.obj` is no longer supported. Use the stub to get objects instead.")

    @staticmethod
    async def _init_existing(client: _Client, existing_app_id: str) -> "_LocalApp":
        # Get all the objects first
        obj_req = api_pb2.AppGetObjectsRequest(app_id=existing_app_id)
        obj_resp = await retry_transient_errors(client.stub.AppGetObjects, obj_req)
        app_page_url = f"https://modal.com/apps/{existing_app_id}"  # TODO (elias): this should come from the backend
        object_ids = {item.tag: item.object.object_id for item in obj_resp.items}
        return _LocalApp(client, existing_app_id, app_page_url, tag_to_object_id=object_ids)

    @staticmethod
    async def _init_new(
        client: _Client,
        description: Optional[str] = None,
        detach: bool = False,
        deploying: bool = False,
        environment_name: str = "",
    ) -> "_LocalApp":
        # Start app
        # TODO(erikbern): maybe this should happen outside of this method?
        app_req = api_pb2.AppCreateRequest(
            description=description,
            initializing=deploying,
            detach=detach,
            environment_name=environment_name,
        )
        app_resp = await retry_transient_errors(client.stub.AppCreate, app_req)
        app_page_url = app_resp.app_logs_url
        logger.debug(f"Created new app with id {app_resp.app_id}")
        return _LocalApp(client, app_resp.app_id, app_page_url, environment_name=environment_name)

    @staticmethod
    async def _init_from_name(
        client: _Client,
        name: str,
        namespace,
        environment_name: str = "",
    ):
        # Look up any existing deployment
        app_req = api_pb2.AppGetByDeploymentNameRequest(
            name=name, namespace=namespace, environment_name=environment_name
        )
        app_resp = await retry_transient_errors(client.stub.AppGetByDeploymentName, app_req)
        existing_app_id = app_resp.app_id or None

        # Grab the app
        if existing_app_id is not None:
            return await _LocalApp._init_existing(client, existing_app_id)
        else:
            return await _LocalApp._init_new(
                client, name, detach=False, deploying=True, environment_name=environment_name
            )

    async def deploy(self, name: str, namespace, object_entity: str) -> str:
        """`App.deploy` is deprecated in favor of `modal.runner.deploy_stub`."""
        deploy_req = api_pb2.AppDeployRequest(
            app_id=self.app_id,
            name=name,
            namespace=namespace,
            object_entity=object_entity,
        )
        try:
            deploy_response = await retry_transient_errors(self._client.stub.AppDeploy, deploy_req)
        except GRPCError as exc:
            if exc.status == Status.INVALID_ARGUMENT:
                raise InvalidError(exc.message)
            if exc.status == Status.FAILED_PRECONDITION:
                raise InvalidError(exc.message)
            raise
        return deploy_response.url

    async def spawn_sandbox(
        self,
        *args,
        **kwargs,
    ):
        """Deprecated. Use `Stub.spawn_sandbox` instead."""
        deprecation_error(date(2023, 9, 11), _LocalApp.spawn_sandbox.__doc__)

    @staticmethod
    async def _deploy_single_object(
        obj: _Object, type_prefix: str, client: _Client, label: str, namespace: int, environment_name: int
    ):
        """mdmd:hidden"""
        app = await _LocalApp._init_from_name(client, label, namespace, environment_name=environment_name)
        existing_object_id: Optional[str] = app._tag_to_object_id.get("_object")
        resolver = Resolver(app._client, environment_name=environment_name, app_id=app.app_id)
        await resolver.load(obj, existing_object_id)
        indexed_object_ids = {"_object": obj.object_id}
        unindexed_object_ids = [obj.object_id for obj in resolver.objects() if obj.object_id is not obj.object_id]
        req_set = api_pb2.AppSetObjectsRequest(
            app_id=app.app_id,
            indexed_object_ids=indexed_object_ids,
            unindexed_object_ids=unindexed_object_ids,
            new_app_state=api_pb2.APP_STATE_UNSPECIFIED,  # app is either already deployed or will be set to deployed after this call
        )
        await retry_transient_errors(client.stub.AppSetObjects, req_set)
        await app.deploy(label, namespace, type_prefix)  # TODO(erikbern): not needed if the app already existed


class _ContainerApp:
    _client: Optional[_Client]
    _app_id: str
    _associated_stub: Optional[Any]  # TODO(erikbern): type
    _environment_name: Optional[str]
    _tag_to_object_id: Dict[str, str]
    _tag_to_handle_metadata: Dict[str, Optional[Message]]

    def __init__(
        self,
        client: _Client,
        app_id: str,
        stub_name: str,
        environment_name: str,
        tag_to_object_id: Dict[str, str],
        tag_to_handle_metadata: Dict[str, Optional[Message]],
    ):
        self._client = client
        self._app_id = app_id
        self._tag_to_object_id = {}
        self._tag_to_handle_metadata = {}
        self._associated_stub = None
        self._stub_name = stub_name
        self._environment_name = environment_name
        self._tag_to_object_id = tag_to_object_id
        self._tag_to_handle_metadata = tag_to_handle_metadata

    @property
    def client(self) -> _Client:
        """A reference to the running App's server client."""
        return self._client

    @property
    def app_id(self) -> str:
        """A unique identifier for this running App."""
        return self._app_id

    def _associate_stub_container(self, stub):
        if self._associated_stub:
            if self._stub_name:
                warning_sub_message = f"stub with the same name ('{self._stub_name}')"
            else:
                warning_sub_message = "unnamed stub"
            logger.warning(
                f"You have more than one {warning_sub_message}. It's recommended to name all your Stubs uniquely when using multiple stubs"
            )
        self._associated_stub = stub

        if stub:
            # Initialize objects on stub
            stub_objects: dict[str, _Object] = dict(stub.get_objects())
            for tag, object_id in self._tag_to_object_id.items():
                obj = stub_objects.get(tag)
                if obj is not None:
                    handle_metadata = self._tag_to_handle_metadata[tag]
                    obj._hydrate(object_id, self._client, handle_metadata)

    def __getitem__(self, tag: str) -> _Object:
        deprecation_error(date(2023, 8, 10), "`app[...]` is no longer supported. Use the stub to get objects instead.")

    def __contains__(self, tag: str) -> bool:
        deprecation_error(
            date(2023, 8, 10), "`obj in app` is no longer supported. Use the stub to get objects instead."
        )

    def __getattr__(self, tag: str) -> _Object:
        if tag.startswith("__"):
            raise AttributeError(f"No such attribute `{tag}`")  # Dumb workaround for doc thing
        deprecation_error(date(2023, 8, 10), "`app.obj` is no longer supported. Use the stub to get objects instead.")

    def _has_object(self, tag: str) -> bool:
        return tag in self._tag_to_object_id

    def _hydrate_object(self, obj, tag: str):
        object_id: str = self._tag_to_object_id[tag]
        metadata: Message = self._tag_to_handle_metadata[tag]
        obj._hydrate(object_id, self._client, metadata)

    def _get_pty(self) -> _Object:
        # TOOD(erikbern): This method has zero tests. It's used in _container_entrypoint
        # Let's try to clean this up ASAP
        object_id = self._tag_to_object_id["_pty_input_stream"]
        metadata = self._tag_to_handle_metadata["_pty_input_stream"]
        return _Object._new_hydrated(object_id, self._client, metadata)

    @staticmethod
    async def init_container(
        client: _Client, app_id: str, stub_name: str = "", environment_name: str = ""
    ) -> "_ContainerApp":
        """Used by the container to bootstrap the app and all its objects. Not intended to be called by Modal users."""
        global _container_app, _is_container_app
        _is_container_app = True

        tag_to_object_id = {}
        tag_to_handle_metadata = {}
        req = api_pb2.AppGetObjectsRequest(app_id=app_id)
        resp = await retry_transient_errors(client.stub.AppGetObjects, req)
        for item in resp.items:
            tag_to_object_id[item.tag] = item.object.object_id
            handle_metadata: Optional[Message] = get_proto_oneof(item.object, "handle_metadata_oneof")
            tag_to_handle_metadata[item.tag] = handle_metadata

        _container_app.__init__(client, app_id, stub_name, environment_name, tag_to_object_id, tag_to_handle_metadata)
        return _container_app

    async def spawn_sandbox(
        self,
        *args,
        **kwargs,
    ):
        """Deprecated. Use `Stub.spawn_sandbox` instead."""
        deprecation_error(date(2023, 9, 11), _ContainerApp.spawn_sandbox.__doc__)

    @staticmethod
    def _reset_container():
        # Just used for tests
        global _is_container_app, _container_app
        _is_container_app = False
        _container_app.__init__(None, "", "", "", {}, {})  # type: ignore


LocalApp = synchronize_api(_LocalApp)
ContainerApp = synchronize_api(_ContainerApp)

_is_container_app = False
_container_app = _ContainerApp(None, "", "", "", {}, {})
container_app = synchronize_api(_container_app)
assert isinstance(container_app, ContainerApp)


def is_local() -> bool:
    """Returns if we are currently on the machine launching/deploying a Modal app

    Returns `True` when executed locally on the user's machine.
    Returns `False` when executed from a Modal container in the cloud.
    """
    return not _is_container_app
