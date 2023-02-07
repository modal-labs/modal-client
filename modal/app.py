# Copyright Modal Labs 2022
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, TypeVar

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors

from ._resolver import Resolver
from .client import _Client
from .config import logger
from .object import Handle, Provider

if TYPE_CHECKING:
    from rich.tree import Tree
else:
    Tree = TypeVar("Tree")


class _App:
    """Apps are the user representation of an actively running Modal process.

    You can obtain an `App` from the `Stub.run()` context manager. While the app
    is running, you can get its `app_id`, `client`, and other useful properties
    from this object.

    ```python
    import modal

    stub = modal.Stub()
    stub.my_secret_object = modal.Secret.from_name("my-secret")

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
        app_page_url: str,
        tag_to_object: Optional[Dict[str, Handle]] = None,
        tag_to_existing_id: Optional[Dict[str, str]] = None,
    ):
        """mdmd:hidden This is the app constructor. Users should not call this directly."""
        self._stub = stub
        self._app_id = app_id
        self._app_page_url = app_page_url
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
        cached_obj = self._local_uuid_to_object.get(obj.local_uuid)
        if cached_obj is not None:
            # We already created this object before, shortcut this method
            return cached_obj

        resolver = Resolver(self, progress, self._client, self.app_id, existing_object_id)

        # Create object
        created_obj = await obj._load(resolver)

        resolver.set_finish()  # finishes any message

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

    async def _create_all_objects(self, progress: Tree, new_app_state: int):  # api_pb2.AppState.V
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
            indexed_object_ids=indexed_object_ids,
            unindexed_object_ids=unindexed_object_ids,
            new_app_state=new_app_state,  # type: ignore
        )
        await retry_transient_errors(self._client.stub.AppSetObjects, req_set)
        return self._tag_to_object

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
        return self._app_page_url

    def __getitem__(self, tag: str) -> Handle:
        # Deprecated?
        return self._tag_to_object[tag]

    def __getattr__(self, tag: str) -> Handle:
        return self._tag_to_object[tag]

    async def _init_container(self, client: _Client, app_id: str):
        self._client = client
        self._app_id = app_id

        req = api_pb2.AppGetObjectsRequest(app_id=app_id)
        resp = await retry_transient_errors(self._client.stub.AppGetObjects, req)
        for item in resp.items:
            obj = Handle._from_id(item.object_id, self._client, item.function)
            self._tag_to_object[item.tag] = obj

    @staticmethod
    async def init_container(client: _Client, app_id: str) -> _App:
        """Used by the container to bootstrap the app and all its objects. Not intended to be called by Modal users."""
        global _container_app, _is_container_app
        _is_container_app = True
        await _container_app._init_container(client, app_id)
        return _container_app

    @staticmethod
    async def _init_existing(stub, client, existing_app_id):
        # Get all the objects first
        obj_req = api_pb2.AppGetObjectsRequest(app_id=existing_app_id)
        obj_resp = await retry_transient_errors(client.stub.AppGetObjects, obj_req)
        app_page_url = f"https://modal.com/apps/{existing_app_id}"  # TODO (elias): this should come from the backend
        object_ids = {item.tag: item.object_id for item in obj_resp.items}
        return _App(stub, client, existing_app_id, app_page_url, tag_to_existing_id=object_ids)

    @staticmethod
    async def _init_new(stub, client, description, detach, deploying) -> "_App":
        # Start app
        # TODO(erikbern): maybe this should happen outside of this method?
        app_req = api_pb2.AppCreateRequest(
            description=description,
            initializing=deploying,
            detach=detach,
        )
        app_resp = await retry_transient_errors(client.stub.AppCreate, app_req)
        app_page_url = app_resp.app_logs_url
        logger.debug(f"Created new app with id {app_resp.app_id}")
        return _App(stub, client, app_resp.app_id, app_page_url)

    @staticmethod
    def _reset_container():
        # Just used for tests
        global _is_container_app, _container_app
        _is_container_app = False
        _container_app.__init__(None, None, None, None)  # type: ignore


App, AioApp = synchronize_apis(_App)

_is_container_app = False
_container_app = _App(None, None, None, None)
container_app, aio_container_app = synchronize_apis(_container_app)
assert isinstance(container_app, App)
assert isinstance(aio_container_app, AioApp)
__doc__container_app = """A reference to the running `modal.App`, accessible from within a running Modal function.
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

    Returns `True` when executed locally on the user's machine.
    Returns `False` when executed from a Modal container in the cloud.
    """
    return not _is_container_app
