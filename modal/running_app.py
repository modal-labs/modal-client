import contextlib
import inspect
import warnings
from typing import Dict, Optional

from rich.tree import Tree

from modal_proto import api_pb2
from modal_utils.async_utils import intercept_coro, synchronize_apis

from ._output import step_completed, step_progress
from .client import _Client
from .config import logger
from .exception import InvalidError, NotFoundError
from .functions import _Function
from .image import _Image
from .object import Object, Ref, ref


async def _lookup_to_id(app_name: str, tag: str, namespace, client: _Client) -> str:
    """Internal method to resolve to an object id."""
    request = api_pb2.AppLookupObjectRequest(
        app_name=app_name,
        object_tag=tag,
        namespace=namespace,
    )
    response = await client.stub.AppLookupObject(request)
    if not response.object_id:
        raise NotFoundError(response.error_message)
    return response.object_id


async def _lookup(
    app_name: str,
    tag: Optional[str] = None,
    namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT,
    client: Optional[_Client] = None,
) -> Object:
    if client is None:
        client = _Client.from_env()
    object_id = await _lookup_to_id(app_name, tag, namespace, client)
    return Object.from_id(object_id, client)


lookup, aio_lookup = synchronize_apis(_lookup)


class _RunningApp:
    _tag_to_object: Dict[str, Object]
    _tag_to_existing_id: Dict[str, str]
    _local_uuid_to_object_id: Dict[str, str]
    _client: _Client
    _app_id: str

    def __init__(
        self,
        app,  # : _App,
        client: _Client,
        app_id: str,
        tag_to_object: Optional[Dict[str, Object]] = None,
        tag_to_existing_id: Optional[Dict[str, str]] = None,
    ):
        self._app = app
        self._app_id = app_id
        self._client = client
        self._tag_to_object = tag_to_object or {}
        self._tag_to_existing_id = tag_to_existing_id or {}
        self._local_uuid_to_object_id = {}

    @property
    def client(self):
        return self._client

    @property
    def app_id(self):
        return self._app_id

    # Method with warning, keep in versions (modal<=0.0.14)
    async def include(self, app_name, tag=None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
        """Looks up an object and return a newly constructed one."""
        warnings.warn("RunningApp.include is deprecated. Use modal.lookup instead", DeprecationWarning)
        return await _lookup(app_name, tag, namespace, self.client)

    @contextlib.contextmanager
    def _progress_ctx(self, progress, obj):
        creating_message = obj._get_creating_message()
        if progress and creating_message:
            step_node = progress.add(step_progress(creating_message))
            yield
            created_message = obj._get_created_message()
            step_node.label = step_completed(created_message, is_substep=True)
        else:
            yield

    async def load(self, obj: Object, progress: Optional[Tree] = None, existing_object_id: Optional[str] = None) -> str:
        """Takes an object as input, create it, and return an object id."""
        if obj.local_uuid in self._local_uuid_to_object_id:
            # We already created this object before, shortcut this method
            return self._local_uuid_to_object_id[obj.local_uuid]

        if isinstance(obj, Ref):
            # TODO: should we just move this code to the Ref class?
            if obj.app_name is not None:
                if obj.definition is not None:
                    from .stub import _Stub

                    _stub = _Stub(obj.app_name)
                    _stub["_object"] = obj.definition
                    await _stub.deploy(client=self._client)
                # A different app
                object_id = await _lookup_to_id(obj.app_name, obj.tag, obj.namespace, self._client)
            else:
                assert not obj.definition
                # Same app
                if obj.tag in self._tag_to_object:
                    object_id = self._tag_to_object[obj.tag].object_id
                else:
                    real_obj = self._app._blueprint[obj.tag]
                    existing_object_id = self._tag_to_existing_id.get(obj.tag)
                    object_id = await self.load(real_obj, progress, existing_object_id)
                    self._tag_to_object[obj.tag] = Object.from_id(object_id, self.client)
        else:

            async def interceptor(awaitable):
                assert isinstance(awaitable, Object)
                return await self.load(awaitable, progress=progress)

            with self._progress_ctx(progress, obj):
                object_id = await intercept_coro(
                    obj._load(self.client, self.app_id, existing_object_id),
                    interceptor,
                )

            if existing_object_id is not None and object_id != existing_object_id:
                # TODO(erikbern): this is a very ugly fix to a problem that's on the server side.
                # Unlike every other object, images are not assigned random ids, but rather an
                # id given by the hash of its contents. This means we can't _force_ an image to
                # have a particular id. The better solution is probably to separate "images"
                # from "image definitions" or something like that, but that's a big project.
                if not existing_object_id.startswith("im-"):
                    raise Exception(
                        f"Tried creating an object using existing id {existing_object_id} but it has id {object_id}"
                    )

        self._local_uuid_to_object_id[obj.local_uuid] = object_id

        if object_id is None:
            raise Exception(f"object_id for object of type {type(obj)} is None")

        return object_id

    async def create_all_objects(self, progress: Tree):
        """Create objects that have been defined but not created on the server."""
        for tag in self._app._blueprint.keys():
            obj = ref(None, tag)
            await self.load(obj, progress)

        # Create the app (and send a list of all tagged obs)
        # TODO(erikbern): we should delete objects from a previous version that are no longer needed
        # We just delete them from the app, but the actual objects will stay around
        object_ids = {tag: obj.object_id for tag, obj in self._tag_to_object.items()}
        req_set = api_pb2.AppSetObjectsRequest(
            app_id=self._app_id,
            object_ids=object_ids,
            client_id=self._client.client_id,
        )
        await self._client.stub.AppSetObjects(req_set)

        # Update all functions client-side to point to the running app
        for obj in self._app._blueprint.values():
            if isinstance(obj, _Function):
                obj.set_local_running_app(self)

    async def disconnect(self):
        # Stop app server-side. This causes:
        # 1. Server to kill any running task
        # 2. Logs to drain (stopping the _get_logs_loop coroutine)
        logger.debug("Stopping the app server-side")
        req_disconnect = api_pb2.AppClientDisconnectRequest(app_id=self._app_id)
        await self._client.stub.AppClientDisconnect(req_disconnect)

    def __getitem__(self, tag: str) -> Object:
        # Deprecated?
        return self._tag_to_object[tag]

    def __getattr__(self, tag: str) -> Object:
        return self._tag_to_object[tag]

    def is_inside(self, image: Optional[Ref] = None):
        # TODO(erikbern): we need a unit test for this
        if image is None:
            image = ref(None, "image")
        elif not isinstance(image, Ref):
            raise InvalidError(
                inspect.cleandoc(
                    """`is_inside` only works for an image associated with an App. For instance:
                app["image"] = DebianSlim()
                if app.is_inside(app["image"]):
                    print("I'm inside!")"""
                )
            )
        if image.tag not in self._tag_to_object:
            # This is some other image, which could belong to some unrelated
            # app or whatever
            return False
        app_image = self._tag_to_object[image.tag]
        assert isinstance(app_image, _Image)
        return app_image._is_inside()

    @staticmethod
    async def init_container(client, app_id, task_id):
        """Used by the container to bootstrap the app and all its objects."""
        # This is a bit of a hacky thing:
        global _container_app, _is_container_app
        _is_container_app = True
        self = _container_app
        self._client = client
        self._app_id = app_id

        req = api_pb2.AppGetObjectsRequest(app_id=app_id, task_id=task_id)
        resp = await self._client.stub.AppGetObjects(req)
        for (
            tag,
            object_id,
        ) in resp.object_ids.items():
            self._tag_to_object[tag] = Object.from_id(object_id, self._client)

        return self

    @staticmethod
    async def init_existing(app, client, existing_app_id):
        # Get all the objects first
        obj_req = api_pb2.AppGetObjectsRequest(app_id=existing_app_id)
        obj_resp = await client.stub.AppGetObjects(obj_req)
        return _RunningApp(app, client, existing_app_id, tag_to_existing_id=dict(obj_resp.object_ids))

    @staticmethod
    async def init_new(app, client, name):
        # Start app
        # TODO(erikbern): maybe this should happen outside of this method?
        app_req = api_pb2.AppCreateRequest(client_id=client.client_id, name=name)
        app_resp = await client.stub.AppCreate(app_req)
        return _RunningApp(app, client, app_resp.app_id)

    @staticmethod
    def reset_container():
        global _is_container_app
        _is_container_app = False


RunningApp, AioRunningApp = synchronize_apis(_RunningApp)

_is_container_app = False
_container_app = _RunningApp(None, None, None)
container_app, aio_container_app = synchronize_apis(_container_app)
assert isinstance(container_app, RunningApp)
assert isinstance(aio_container_app, AioRunningApp)


def is_local() -> bool:
    """Returns whether we're running in the cloud or not."""
    return not _is_container_app
