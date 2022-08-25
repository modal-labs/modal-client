import contextlib
from typing import Dict, Optional, Union

from rich.tree import Tree

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from ._output import step_completed, step_progress
from .client import _Client
from .config import logger
from .exception import NotFoundError
from .functions import _Function
from .image import _Image
from .object import LocalRef, Object, PersistedRef, RemoteRef


async def _lookup(
    app_name: str,
    tag: Optional[str] = None,
    namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT,
    client: Optional[_Client] = None,
) -> Object:
    """Returns a handle to a tagged object in a deployment on Modal."""
    if client is None:
        client = await _Client.from_env()
    request = api_pb2.AppLookupObjectRequest(
        app_name=app_name,
        object_tag=tag,
        namespace=namespace,
    )
    response = await client.stub.AppLookupObject(request)
    if not response.object_id:
        raise NotFoundError(response.error_message)
    obj = Object.from_id(response.object_id, client)
    if isinstance(obj, _Function):
        # TODO(erikbern): treating this as a special case right now, but we should generalize it
        obj.initialize_from_proto(response.function)
    return obj


lookup, aio_lookup = synchronize_apis(_lookup)


class _App:
    """Apps are the user representation of an actively running Modal process.

    You can obtain an `App` from the `Stub.run()` context manager. While the app
    is running, you can get its `app_id`, `client`, and other useful properties
    from this object.

    ```
    import modal

    stub = modal.Stub()
    stub.my_secret_object = modal.ref("my-secret")

    if __name__ == "__main__":
        with stub.run() as app:
            print(app.client)
            print(app.app_id)
            print(app.my_secret_object)
    ```
    """

    _tag_to_object: Dict[str, Object]
    _tag_to_existing_id: Dict[str, str]
    _local_uuid_to_object: Dict[str, Object]
    _client: _Client
    _app_id: str

    def __init__(
        self,
        stub,  # : _Stub,
        client: _Client,
        app_id: str,
        tag_to_object: Optional[Dict[str, Object]] = None,
        tag_to_existing_id: Optional[Dict[str, str]] = None,
    ):
        """mdmd:hidden This is the app constructor. Users should not call this directly."""
        self._stub = stub
        self._app_id = app_id
        self._client = client
        self._tag_to_object = tag_to_object or {}
        self._tag_to_existing_id = tag_to_existing_id or {}
        self._local_uuid_to_object = {}

    @property
    def client(self):
        return self._client

    @property
    def app_id(self):
        return self._app_id

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

    async def load(
        self, obj: Object, progress: Optional[Tree] = None, existing_object_id: Optional[str] = None
    ) -> Object:
        """Send a server request to create an object in this app, and return its ID."""
        if obj.local_uuid in self._local_uuid_to_object:
            # We already created this object before, shortcut this method
            return self._local_uuid_to_object[obj.local_uuid]

        # TODO: should we just move most of this code to the Ref classes?
        if isinstance(obj, PersistedRef):
            from .stub import _Stub

            _stub = _Stub(obj.app_name)
            _stub["_object"] = obj.definition
            await _stub.deploy(client=self._client)
            created_obj = await _lookup(obj.app_name, client=self._client)

        elif isinstance(obj, RemoteRef):
            created_obj = await _lookup(obj.app_name, obj.tag, obj.namespace, client=self._client)

        elif isinstance(obj, LocalRef):
            if obj.tag in self._tag_to_object:
                created_obj = self._tag_to_object[obj.tag]
            else:
                real_obj = self._stub._blueprint[obj.tag]
                existing_object_id = self._tag_to_existing_id.get(obj.tag)
                created_obj = await self.load(real_obj, progress, existing_object_id)
                self._tag_to_object[obj.tag] = created_obj
        else:

            async def loader(obj: Object) -> str:
                assert isinstance(obj, Object)
                created_obj = await self.load(obj, progress=progress)
                return created_obj.object_id

            with self._progress_ctx(progress, obj):
                object_id = await obj._load(self.client, self.app_id, loader, existing_object_id)

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

            created_obj = Object.from_id(object_id, self.client)

        self._local_uuid_to_object[obj.local_uuid] = created_obj
        return created_obj

    async def create_all_objects(self, progress: Tree):
        """Create objects that have been defined but not created on the server."""
        for tag in self._stub._blueprint.keys():
            obj = LocalRef(tag)
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
        for obj in self._stub._blueprint.values():
            if isinstance(obj, _Function):
                obj.set_local_app(self)

    async def disconnect(self) -> None:
        """Tell the server to stop this app, terminating all running tasks."""
        logger.debug("Sending app disconnect request")
        req_disconnect = api_pb2.AppClientDisconnectRequest(app_id=self._app_id)
        await self._client.stub.AppClientDisconnect(req_disconnect)

    def __getitem__(self, tag: str) -> Object:
        # Deprecated?
        return self._tag_to_object[tag]

    def __getattr__(self, tag: str) -> Object:
        return self._tag_to_object[tag]

    def _is_inside(self, image: Union[LocalRef, _Image]) -> bool:
        if isinstance(image, LocalRef):
            if image.tag not in self._tag_to_object:
                # This is some other image, which could belong to some unrelated
                # app or whatever
                return False
            app_image = self._tag_to_object[image.tag]
        else:
            app_image = image
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
    async def init_existing(stub, client, existing_app_id):
        # Get all the objects first
        obj_req = api_pb2.AppGetObjectsRequest(app_id=existing_app_id)
        obj_resp = await client.stub.AppGetObjects(obj_req)
        return _App(stub, client, existing_app_id, tag_to_existing_id=dict(obj_resp.object_ids))

    @staticmethod
    async def init_new(stub, client, description):
        # Start app
        # TODO(erikbern): maybe this should happen outside of this method?
        app_req = api_pb2.AppCreateRequest(client_id=client.client_id, description=description)
        app_resp = await client.stub.AppCreate(app_req)
        logger.debug(f"Created new app with id {app_resp.app_id}")
        return _App(stub, client, app_resp.app_id)

    @staticmethod
    def reset_container():
        global _is_container_app
        _is_container_app = False


App, AioApp = synchronize_apis(_App)

_is_container_app = False
_container_app = _App(None, None, None)
container_app, aio_container_app = synchronize_apis(_container_app)
assert isinstance(container_app, App)
assert isinstance(aio_container_app, AioApp)


def is_local() -> bool:
    """Returns whether we're running in the cloud or not."""
    return not _is_container_app
