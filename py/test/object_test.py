# Copyright Modal Labs 2022
import pytest
from contextlib import asynccontextmanager

from modal import Secret
from modal._load_context import LoadContext
from modal._object import _Object
from modal.dict import Dict, _Dict
from modal.exception import InvalidError
from modal.queue import _Queue
from modal_proto import api_pb2
from test.conftest import servicer_factory


def test_new_hydrated(client):
    assert isinstance(_Dict._new_hydrated("di-123", client, None), _Dict)
    assert isinstance(_Queue._new_hydrated("qu-123", client, None), _Queue)

    with pytest.raises(InvalidError):
        _Queue._new_hydrated("di-123", client, None)  # Wrong prefix for type

    assert isinstance(_Object._new_hydrated("qu-123", client, None), _Queue)
    assert isinstance(_Object._new_hydrated("di-123", client, None), _Dict)

    with pytest.raises(InvalidError):
        _Object._new_hydrated("xy-123", client, None)


def test_on_demand_hydration(client):
    obj = Dict.from_name("test-dict", create_if_missing=True).hydrate(client)
    assert obj.object_id is not None


def test_constructor():
    with pytest.raises(InvalidError) as excinfo:
        Secret({"foo": 123})

    assert "Secret" in str(excinfo.value)
    assert "constructor" in str(excinfo.value)


def test_types():
    assert _Object._get_type_from_id("di-123") == _Dict
    assert _Dict._is_id_type("di-123")
    assert not _Dict._is_id_type("qu-123")
    assert _Queue._is_id_type("qu-123")
    assert not _Queue._is_id_type("di-123")


@pytest.mark.asyncio
async def test_object_rehydration_new_hydrated(blob_server, container_env):
    import modal.client

    @asynccontextmanager
    async def fresh_servicer_client(credentials):
        async with servicer_factory(blob_server, credentials) as servicer:
            async with modal.client._Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client:
                yield servicer, client

    pre_snapshot_credentials = ("ak-123", "as-123")
    async with fresh_servicer_client(pre_snapshot_credentials) as (servicer, client):
        obj = _Object._new_hydrated("qu-123", client, None)
        await obj.hydrate()  # "bare" hydrate happens for all "live_method" style methods
        assert obj.client is client
        await obj.hydrate()  # this shouldn't do anything
        assert obj.client is client
        # simulate snapshot
        await client._close(prep_for_restore=True)
        await obj.hydrate()  # This previously raised an error since obj lacks a loader
        assert obj.client is not client  # client should have been replaced
        post_rehydration_client = obj.client
        await obj.hydrate()
        assert obj.client is post_rehydration_client  # should not create a new client *again*


@pytest.mark.asyncio
async def test_object_rehydration_loader(blob_server, container_env):
    import modal.client

    @asynccontextmanager
    async def fresh_servicer_client(credentials):
        async with servicer_factory(blob_server, credentials) as servicer:
            async with modal.client._Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client:
                yield servicer, client

    pre_snapshot_credentials = ("ak-123", "as-123")
    async with fresh_servicer_client(pre_snapshot_credentials) as (servicer, client):
        hydration_i = 0

        async def _load(self: _Object, resolver, load_context, existing_object_id):
            nonlocal hydration_i
            self._dummy_metadata = hydration_i  # type: ignore
            hydration_i += 1
            self._hydrate("qu-123", load_context.client, None)

        obj = _Queue._from_loader(
            _load, "custom-queue", load_context_overrides=LoadContext(client=client), hydrate_lazily=True
        )
        assert obj._client is None
        await obj.hydrate()  # "bare" hydrate happens for all "live_method" style methods
        assert obj._dummy_metadata == 0  # type: ignore
        assert obj.client is client  # taken from the load_context
        await obj.hydrate()  # this shouldn't do anything
        assert obj._dummy_metadata == 0  # type: ignore
        assert obj.client is client  # still the load context client
        # simulate snapshot
        await client._close(prep_for_restore=True)
        await obj.hydrate()  # This should call the loader again to get fresh metadata
        assert obj._dummy_metadata == 1  # type: ignore
        assert obj.client is not client  # client should have been replaced
        rehydation_client = obj.client
        await obj.hydrate()  # This should not be reloading - wasteful!
        assert obj._dummy_metadata == 1  # type: ignore
        assert obj.client is rehydation_client
