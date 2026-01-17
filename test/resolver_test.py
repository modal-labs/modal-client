# Copyright Modal Labs 2023
import asyncio
import pytest
import time
from typing import Optional, cast

import modal.client
from modal._load_context import LoadContext
from modal._object import _Object
from modal._resolver import Resolver
from modal._utils.async_utils import TaskContext, synchronizer
from modal.exception import NotFoundError


@pytest.mark.flaky(max_runs=2)
@pytest.mark.asyncio
async def test_multi_resolve_sequential_loads_once(client):
    _client = cast(modal.client._Client, synchronizer._translate_in(client))
    resolver = Resolver()

    load_count = 0

    class _DumbObject(_Object, type_prefix="zz"):
        pass

    async def _load(
        self: _DumbObject, resolver: Resolver, load_context: LoadContext, existing_object_id: Optional[str]
    ):
        nonlocal load_count
        load_count += 1
        self._hydrate("zz-123", load_context.client, None)
        await asyncio.sleep(0.1)

    obj = _DumbObject._from_loader(_load, "DumbObject()", load_context_overrides=LoadContext.empty())

    t0 = time.monotonic()
    load_context = LoadContext(client=_client)
    await resolver.load(obj, load_context)
    await resolver.load(obj, load_context)
    assert 0.08 < time.monotonic() - t0 < 0.15

    assert load_count == 1


@pytest.mark.asyncio
async def test_multi_resolve_concurrent_loads_once(client):
    _client = cast(modal.client._Client, synchronizer._translate_in(client))
    resolver = Resolver()

    load_count = 0

    class _DumbObject(_Object, type_prefix="zz"):
        pass

    async def _load(
        self: _DumbObject, resolver: Resolver, load_context: LoadContext, existing_object_id: Optional[str]
    ):
        nonlocal load_count
        load_count += 1
        self._hydrate("zz-123", load_context.client, None)
        await asyncio.sleep(0.1)

    obj = _DumbObject._from_loader(_load, "DumbObject()", load_context_overrides=LoadContext.empty())
    t0 = time.monotonic()
    load_context = LoadContext(client=_client)
    await asyncio.gather(resolver.load(obj, load_context), resolver.load(obj, load_context))
    assert 0.08 < time.monotonic() - t0 < 0.17
    assert load_count == 1


def test_resolver_without_rich(no_rich):
    resolver = Resolver()
    resolver.add_status_row()
    with resolver.display():
        pass


@pytest.mark.asyncio
async def test_resolver_shared_dependency_exception_priority(client):
    """
    Test that real exceptions are raised (not CancelledError) when loading shared dependencies.

    This test simulates a scenario where:
    1. Two objects (func_a, func_b) share a common dependency (shared_volume)
    2. func_a ALSO depends on a failing_secret that raises NotFoundError
    3. When loading both functions concurrently:
       - func_a's gather(shared_volume, failing_secret) will cancel the shared_volume
         task when failing_secret raises
       - func_b is also awaiting the same shared_volume task (via resolver caching)
       - func_b may see CancelledError from the shared task before func_a's exception propagates

    The Resolver (or TaskContext.gather) should ensure the real exception (NotFoundError)
    is raised, not CancelledError.
    """
    _client = cast(modal.client._Client, synchronizer._translate_in(client))
    resolver = Resolver()
    load_context = LoadContext(client=_client)

    # A "Volume" object that takes time to load (simulates slow dependency)
    class _SharedVolume(_Object, type_prefix="vo"):
        pass

    async def _load_volume(
        self: _SharedVolume, resolver: Resolver, load_context: LoadContext, existing_object_id: Optional[str]
    ):
        await asyncio.sleep(0.5)  # Slow enough to still be loading when sibling fails
        self._hydrate("vo-shared123", load_context.client, None)

    shared_volume = _SharedVolume._from_loader(
        _load_volume, "Volume(shared)", load_context_overrides=LoadContext.empty()
    )

    # A "Secret" object that fails to load
    class _FailingSecret(_Object, type_prefix="st"):
        pass

    async def _load_failing_secret(
        self: _FailingSecret, resolver: Resolver, load_context: LoadContext, existing_object_id: Optional[str]
    ):
        raise NotFoundError("Secret 'missing-secret' not found")

    failing_secret = _FailingSecret._from_loader(
        _load_failing_secret, "Secret(missing)", load_context_overrides=LoadContext.empty()
    )

    # func_a depends on BOTH the shared volume and the failing secret
    class _FunctionA(_Object, type_prefix="fu"):
        pass

    async def _load_func_a(
        self: _FunctionA, resolver: Resolver, load_context: LoadContext, existing_object_id: Optional[str]
    ):
        # Load both dependencies - one will fail
        await TaskContext.gather(
            resolver.load(shared_volume, load_context),
            resolver.load(failing_secret, load_context),
        )
        self._hydrate("fu-funca123", load_context.client, None)

    func_a = _FunctionA._from_loader(_load_func_a, "Function(a)", load_context_overrides=LoadContext.empty())

    # func_b depends ONLY on the shared volume
    class _FunctionB(_Object, type_prefix="fu"):
        pass

    async def _load_func_b(
        self: _FunctionB, resolver: Resolver, load_context: LoadContext, existing_object_id: Optional[str]
    ):
        await resolver.load(shared_volume, load_context)
        self._hydrate("fu-funcb123", load_context.client, None)

    func_b = _FunctionB._from_loader(_load_func_b, "Function(b)", load_context_overrides=LoadContext.empty())

    # Load both functions concurrently - this is where the race happens
    with pytest.raises(NotFoundError) as exc_info:
        await TaskContext.gather(
            resolver.load(func_a, load_context),
            resolver.load(func_b, load_context),
        )

    # Verify the correct exception was raised (not CancelledError)
    assert "missing-secret" in str(exc_info.value), f"Got unexpected error: {exc_info.value}"
