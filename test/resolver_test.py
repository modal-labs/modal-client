# Copyright Modal Labs 2023
import asyncio
import pytest
import time
from typing import Optional

from modal._output import OutputManager
from modal._resolver import Resolver
from modal.object import _Object


@pytest.mark.flaky(max_runs=2)
@pytest.mark.asyncio
async def test_multi_resolve_sequential_loads_once():
    output_manager = OutputManager(None, show_progress=False)
    resolver = Resolver(None, output_mgr=output_manager, environment_name="", app_id=None)

    load_count = 0

    class _DumbObject(_Object, type_prefix="zz"):
        pass

    async def _load(self: _DumbObject, resolver: Resolver, existing_object_id: Optional[str]):
        nonlocal load_count
        load_count += 1
        self._hydrate("zz-123", resolver.client, None)
        await asyncio.sleep(0.1)

    obj = _DumbObject._from_loader(_load, "DumbObject()")

    t0 = time.monotonic()
    await resolver.load(obj)
    await resolver.load(obj)
    assert 0.08 < time.monotonic() - t0 < 0.15

    assert load_count == 1


@pytest.mark.asyncio
async def test_multi_resolve_concurrent_loads_once():
    output_manager = OutputManager(None, show_progress=False)
    resolver = Resolver(None, output_mgr=output_manager, environment_name="", app_id=None)

    load_count = 0

    class _DumbObject(_Object, type_prefix="zz"):
        pass

    async def _load(self: _DumbObject, resolver: Resolver, existing_object_id: Optional[str]):
        nonlocal load_count
        load_count += 1
        self._hydrate("zz-123", resolver.client, None)
        await asyncio.sleep(0.1)

    obj = _DumbObject._from_loader(_load, "DumbObject()")
    t0 = time.monotonic()
    await asyncio.gather(resolver.load(obj), resolver.load(obj))
    assert 0.08 < time.monotonic() - t0 < 0.17
    assert load_count == 1
