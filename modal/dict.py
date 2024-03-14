# Copyright Modal Labs 2022
from typing import Any, Optional

from modal_proto import api_pb2

from ._resolver import Resolver
from ._serialization import deserialize, serialize
from ._types import typechecked
from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import retry_transient_errors
from .client import _Client
from .config import logger
from .exception import deprecation_error, deprecation_warning
from .object import _get_environment_name, _Object, live_method


def _serialize_dict(data):
    return [api_pb2.DictEntry(key=serialize(k), value=serialize(v)) for k, v in data.items()]


class _Dict(_Object, type_prefix="di"):
    """Distributed dictionary for storage in Modal apps.

    Keys and values can be essentially any object, so long as they can be
    serialized by `cloudpickle`, including Modal objects.

    **Lifetime of a Dict and its items**

    A `Dict` matches the lifetime of the app it is attached to, but an individual entry will expire 30 days after it was last added to its Dict object.
    Because of this, `Dict`s are best not used for
    long-term storage. All data is deleted when the app is stopped.

    **Usage**

    Create a new `Dict` with `Dict.new()`, then assign it to a stub or function.

    ```python
    from modal import Dict, Stub

    stub = Stub()
    stub.my_dict = Dict.new()

    @stub.local_entrypoint()
    def main():
        stub.my_dict["some key"] = "some value"
        stub.my_dict[123] = 456

        assert stub.my_dict["some key"] == "some value"
        assert stub.my_dict[123] == 456
    ```

    For more examples, see the [guide](/docs/guide/dicts-and-queues#modal-dicts).
    """

    @typechecked
    @staticmethod
    def new(data: Optional[dict] = None) -> "_Dict":
        """Create a new Dict, optionally with initial data."""

        async def _load(self: _Dict, resolver: Resolver, existing_object_id: Optional[str]):
            serialized = _serialize_dict(data if data is not None else {})
            req = api_pb2.DictCreateRequest(
                app_id=resolver.app_id, data=serialized, existing_dict_id=existing_object_id
            )
            response = await resolver.client.stub.DictCreate(req)
            logger.debug(f"Created dict with id {response.dict_id}")
            self._hydrate(response.dict_id, resolver.client, None)

        return _Dict._from_loader(_load, "Dict()", is_another_app=True)

    def __init__(self, data={}):
        """mdmd:hidden"""
        deprecation_error((2023, 6, 27), "`Dict({...})` is deprecated. Please use `Dict.new({...})` instead.")
        obj = _Dict.new(data)
        self._init_from_other(obj)

    @staticmethod
    def from_name(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> "_Dict":
        """Create a reference to a persisted Dict

        **Examples**

        ```python notest
        # In one app:
        stub.dict = Dict.persisted("my-dict")

        # Later, in another app or Python file:
        stub.dict = Dict.from_name("my-dict")
        ```
        """

        async def _load(self: _Dict, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.DictGetOrCreateRequest(
                deployment_name=label,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
                object_creation_type=(api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING if create_if_missing else None),
            )
            response = await resolver.client.stub.DictGetOrCreate(req)
            logger.debug(f"Created dict with id {response.dict_id}")
            self._hydrate(response.dict_id, resolver.client, None)

        return _Dict._from_loader(_load, "Dict()")

    @staticmethod
    def persisted(
        label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, environment_name: Optional[str] = None
    ) -> "_Dict":
        """Deprecated! Use `Dict.from_name(name, create_if_missing=True)`."""
        deprecation_warning((2024, 3, 1), _Dict.persisted.__doc__)
        return _Dict.from_name(label, namespace, environment_name, create_if_missing=True)

    @staticmethod
    async def lookup(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> "_Dict":
        """Lookup a dict with a given name and tag.

        ```python
        d = modal.Dict.lookup("my-dict")
        d["xyz"] = 123
        ```
        """
        obj = _Dict.from_name(
            label, namespace=namespace, environment_name=environment_name, create_if_missing=create_if_missing
        )
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client)
        await resolver.load(obj)
        return obj

    @live_method
    async def clear(self) -> None:
        """Remove all items from the modal.Dict."""
        req = api_pb2.DictClearRequest(dict_id=self.object_id)
        await retry_transient_errors(self._client.stub.DictClear, req)

    @live_method
    async def get(self, key: Any, default: Optional[Any] = None) -> Any:
        """Get the value associated with a key.

        Returns `default` if key does not exist.
        """
        req = api_pb2.DictGetRequest(dict_id=self.object_id, key=serialize(key))
        resp = await retry_transient_errors(self._client.stub.DictGet, req)
        if not resp.found:
            return default
        return deserialize(resp.value, self._client)

    @live_method
    async def contains(self, key: Any) -> bool:
        """Return if a key is present."""
        req = api_pb2.DictContainsRequest(dict_id=self.object_id, key=serialize(key))
        resp = await retry_transient_errors(self._client.stub.DictContains, req)
        return resp.found

    @live_method
    async def len(self) -> int:
        """Return the length of the dictionary, including any expired keys."""
        req = api_pb2.DictLenRequest(dict_id=self.object_id)
        resp = await retry_transient_errors(self._client.stub.DictLen, req)
        return resp.len

    @live_method
    async def __getitem__(self, key: Any) -> Any:
        """Get the value associated with a key.

        This function only works in a synchronous context.
        """
        NOT_FOUND = object()
        value = await self.get(key, NOT_FOUND)
        if value is NOT_FOUND:
            raise KeyError(f"{key} not in dict {self.object_id}")

        return value

    @live_method
    async def update(self, **kwargs) -> None:
        """Update the dictionary with additional items."""
        serialized = _serialize_dict(kwargs)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await retry_transient_errors(self._client.stub.DictUpdate, req)

    @live_method
    async def put(self, key: Any, value: Any) -> None:
        """Add a specific key-value pair to the dictionary."""
        updates = {key: value}
        serialized = _serialize_dict(updates)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await retry_transient_errors(self._client.stub.DictUpdate, req)

    @live_method
    async def __setitem__(self, key: Any, value: Any) -> None:
        """Set a specific key-value pair to the dictionary.

        This function only works in a synchronous context.
        """
        return await self.put(key, value)

    @live_method
    async def pop(self, key: Any) -> Any:
        """Remove a key from the dictionary, returning the value if it exists."""
        req = api_pb2.DictPopRequest(dict_id=self.object_id, key=serialize(key))
        resp = await retry_transient_errors(self._client.stub.DictPop, req)
        if not resp.found:
            raise KeyError(f"{key} not in dict {self.object_id}")
        return deserialize(resp.value, self._client)

    @live_method
    async def __delitem__(self, key: Any) -> Any:
        """Delete a key from the dictionary.

        This function only works in a synchronous context.
        """
        return await self.pop(key)

    @live_method
    async def __contains__(self, key: Any) -> bool:
        """Return if a key is present.

        This function only works in a synchronous context.
        """
        return await self.contains(key)


Dict = synchronize_api(_Dict)
