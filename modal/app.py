from typing import Dict, Optional, Union

from rich.tree import Tree

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from ._output import step_progress
from .client import _Client
from .config import logger
from .exception import NotFoundError
from .functions import _FunctionHandle
from .image import _ImageHandle
from .object import Handle, LocalRef, PersistedRef, Provider, RemoteRef


async def _lookup(
    app_name: str,
    tag: Optional[str] = None,
    namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT,
    client: Optional[_Client] = None,
) -> Handle:
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
    obj = Handle._from_id(response.object_id, client)
    if isinstance(obj, _FunctionHandle):
        # TODO(erikbern): treating this as a special case right now, but we should generalize it
        obj._initialize_from_proto(response.function)
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

    _tag_to_object: Dict[str, Handle]
    _tag_to_existing_id: Dict[str, str]
    _local_uuid_to_object: Dict[str, Handle]
    _client: _Client
    _app_id: str

    def __init__(
        self,
        stub,  # : _Stub,
        client: _Client,
        app_id: str,
        tag_to_object: Optional[Dict[str, Handle]] = None,
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
    def client(self) -> _Client:
        """A reference to the running App's server client."""
        return self._client

    @property
    def app_id(self) -> str:
        """A unique identifier for this running App."""
        return self._app_id

    async def _load(
        self, obj: Provider, progress: Optional[Tree] = None, existing_object_id: Optional[str] = None
    ) -> Handle:
        """Send a server request to create an object in this app, and return its ID."""
        cached_obj = self._load_cached(obj)
        if cached_obj is not None:
            # We already created this object before, shortcut this method
            return cached_obj

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
                created_obj = await self._load(real_obj, progress, existing_object_id)
                self._tag_to_object[obj.tag] = created_obj
        else:

            async def loader(obj: Provider) -> str:
                assert isinstance(obj, Provider)
                created_obj = await self._load(obj, progress=progress)
                assert isinstance(created_obj, Handle)
                return created_obj.object_id

            step_node = None

            def set_message(message):
                nonlocal step_node
                if progress:
                    if step_node is None:
                        step_node = progress.add(step_progress(message))
                    else:
                        step_node.label = message

            # Create object
            created_obj = await obj._load(self.client, self.app_id, loader, set_message, existing_object_id)

            if existing_object_id is not None and created_obj.object_id != existing_object_id:
                # TODO(erikbern): this is a very ugly fix to a problem that's on the server side.
                # Unlike every other object, images are not assigned random ids, but rather an
                # id given by the hash of its contents. This means we can't _force_ an image to
                # have a particular id. The better solution is probably to separate "images"
                # from "image definitions" or something like that, but that's a big project.
                if not existing_object_id.startswith("im-"):
                    raise Exception(
                        f"Tried creating an object using existing id {existing_object_id}"
                        f" but it has id {created_obj.object_id}"
                    )

        self._local_uuid_to_object[obj.local_uuid] = created_obj
        return created_obj

    def _load_cached(self, obj: Provider) -> Optional[Handle]:
        """Try to load a previously-loaded object, without making network requests.

        Returns `None` if the object has not been previously loaded.
        """
        return self._local_uuid_to_object.get(obj.local_uuid)

    async def _create_all_objects(self, progress: Tree):
        """Create objects that have been defined but not created on the server."""
        for tag in self._stub._blueprint.keys():
            obj: Provider = LocalRef(tag)
            await self._load(obj, progress)

        # Create the app (and send a list of all tagged obs)
        # TODO(erikbern): we should delete objects from a previous version that are no longer needed
        # We just delete them from the app, but the actual objects will stay around
        indexed_object_ids = {tag: obj.object_id for tag, obj in self._tag_to_object.items()}
        unindexed_object_ids = list(
            set(obj.object_id for obj in self._local_uuid_to_object.values())
            - set(obj.object_id for obj in self._tag_to_object.values())
        )
        req_set = api_pb2.AppSetObjectsRequest(
            app_id=self._app_id,
            client_id=self._client.client_id,
            indexed_object_ids=indexed_object_ids,
            unindexed_object_ids=unindexed_object_ids,
        )
        await self._client.stub.AppSetObjects(req_set)
        return self._tag_to_object

    async def disconnect(self):
        """Tell the server to stop this app, terminating all running tasks."""
        logger.debug("Sending app disconnect request")
        req_disconnect = api_pb2.AppClientDisconnectRequest(app_id=self._app_id)
        await self._client.stub.AppClientDisconnect(req_disconnect)

    def __getitem__(self, tag: str) -> Handle:
        # Deprecated?
        return self._tag_to_object[tag]

    def __getattr__(self, tag: str) -> Handle:
        return self._tag_to_object[tag]

    def _is_inside(self, image: Union[LocalRef, _ImageHandle]) -> bool:
        if isinstance(image, LocalRef):
            if image.tag not in self._tag_to_object:
                # This is some other image, which could belong to some unrelated
                # app or whatever
                return False
            app_image = self._tag_to_object[image.tag]
        else:
            app_image = image
        assert isinstance(app_image, _ImageHandle)
        return app_image._is_inside()

    @staticmethod
    async def _init_container(client, app_id, task_id):
        """Used by the container to bootstrap the app and all its objects."""
        # This is a bit of a hacky thing:
        global _container_app, _is_container_app
        _is_container_app = True
        self = _container_app
        self._client = client
        self._app_id = app_id

        req = api_pb2.AppGetObjectsRequest(app_id=app_id, task_id=task_id)
        resp = await self._client.stub.AppGetObjects(req)
        for item in resp.items:
            obj = Handle._from_id(item.object_id, self._client)
            if isinstance(obj, _FunctionHandle):
                # TODO(erikbern): treating this as a special case right now, but we should generalize it
                obj._initialize_from_proto(item.function)
            self._tag_to_object[item.tag] = obj

        if "image" not in self._tag_to_object:
            from .stub import _default_image

            await self._load(_default_image)

        return self

    @staticmethod
    async def _init_existing(stub, client, existing_app_id):
        # Get all the objects first
        obj_req = api_pb2.AppGetObjectsRequest(app_id=existing_app_id)
        obj_resp = await client.stub.AppGetObjects(obj_req)
        return _App(stub, client, existing_app_id, tag_to_existing_id=dict(obj_resp.object_ids))

    @staticmethod
    async def _init_new(stub, client, description):
        # Start app
        # TODO(erikbern): maybe this should happen outside of this method?
        app_req = api_pb2.AppCreateRequest(client_id=client.client_id, description=description)
        app_resp = await client.stub.AppCreate(app_req)
        logger.debug(f"Created new app with id {app_resp.app_id}")
        return _App(stub, client, app_resp.app_id)

    @staticmethod
    def _reset_container():
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
