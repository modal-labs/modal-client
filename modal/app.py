from typing import Dict, Optional

from rich.tree import Tree

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors

from ._output import step_completed, step_progress, step_progress_update
from .client import _Client
from .config import logger
from .functions import _FunctionHandle
from .object import Handle, Provider


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
        app_logs_url: str,
        tag_to_object: Optional[Dict[str, Handle]] = None,
        tag_to_existing_id: Optional[Dict[str, str]] = None,
    ):
        """mdmd:hidden This is the app constructor. Users should not call this directly."""
        self._stub = stub
        self._app_id = app_id
        self._app_logs_url = app_logs_url
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

        async def loader(obj: Provider) -> str:
            assert isinstance(obj, Provider)
            created_obj = await self._load(obj, progress=progress)
            assert isinstance(created_obj, Handle)
            return created_obj.object_id

        last_message, spinner, step_node = None, None, None

        def set_message(message):
            nonlocal last_message, spinner, step_node
            last_message = message
            if progress:
                if step_node is None:
                    spinner = step_progress()
                    step_node = progress.add(spinner)
                step_progress_update(spinner, message)

        # Create object
        created_obj = await obj._load(self.client, self.app_id, loader, set_message, existing_object_id)

        # Change message to a completed step
        if progress and last_message:
            step_node.label = step_completed(last_message, is_substep=True)

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
        for tag, provider in self._stub._blueprint.items():
            existing_object_id = self._tag_to_existing_id.get(tag)
            self._tag_to_object[tag] = await self._load(provider, progress, existing_object_id)

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
        logger.debug("Sending app disconnect/stop request")
        req_disconnect = api_pb2.AppClientDisconnectRequest(app_id=self._app_id)
        await retry_transient_errors(self._client.stub.AppClientDisconnect, req_disconnect)

    def log_url(self):
        return self._app_logs_url

    def __getitem__(self, tag: str) -> Handle:
        # Deprecated?
        return self._tag_to_object[tag]

    def __getattr__(self, tag: str) -> Handle:
        return self._tag_to_object[tag]

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
        resp = await retry_transient_errors(self._client.stub.AppGetObjects, req)
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
        obj_resp = await retry_transient_errors(client.stub.AppGetObjects, obj_req)
        app_logs_url = f"https://modal.com/apps/{existing_app_id}"  # TODO (elias): this should come from the backend
        return _App(stub, client, existing_app_id, app_logs_url, tag_to_existing_id=dict(obj_resp.object_ids))

    @staticmethod
    async def _init_new(stub, client, description, detach) -> "_App":
        # Start app
        # TODO(erikbern): maybe this should happen outside of this method?
        app_req = api_pb2.AppCreateRequest(client_id=client.client_id, description=description, detach=detach)
        app_resp = await retry_transient_errors(client.stub.AppCreate, app_req)
        logger.debug(f"Created new app with id {app_resp.app_id}")
        return _App(stub, client, app_resp.app_id, app_resp.app_logs_url)

    @staticmethod
    def _reset_container():
        global _is_container_app
        _is_container_app = False


App, AioApp = synchronize_apis(_App)

_is_container_app = False
_container_app = _App(None, None, None, None)
container_app, aio_container_app = synchronize_apis(_container_app)
assert isinstance(container_app, App)
assert isinstance(aio_container_app, AioApp)
__doc__container_app = """A reference to the running modal.App, accessible from within a running Modal function.
Useful for accessing object handles for any Modal objects declared on the stub, e.g:

```python
stub = modal.Stub()
stub.data = modal.Dict()

@stub.function
def store_something(key, value):
    data: modal.DictHandle = modal.container_app.data
    data.put(key, value)
```
"""


def is_local() -> bool:
    """Returns if we are currently on the machine launching/deploying a Modal app

    Returns True when executed locally on the user's machine.
    Returns False when executed from a Modal container in the cloud.
    """
    return not _is_container_app
