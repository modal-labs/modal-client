# Copyright Modal Labs 2023
import asyncio
import pytest
import time
from typing import Optional

from modal._output import OutputManager
from modal._resolver import Resolver
from modal.object import _Handle, _Provider


@pytest.mark.asyncio
async def test_multi_resolve_sequential_loads_once():
    output_manager = OutputManager(None, show_progress=False)
    resolver = Resolver(output_manager, client=None, environment_name="", app_id=None)

    load_count = 0

    class _DumbHandle(_Handle):
        pass

    class _DumbProvider(_Provider[_DumbHandle]):
        pass

    async def _load(resolver: Resolver, existing_object_id: Optional[str], handle: _DumbHandle):
        nonlocal load_count
        load_count += 1
        await asyncio.sleep(0.1)

    obj = _DumbProvider._from_loader(_load, "DumbProvider()")

    t0 = time.monotonic()
    await resolver.load(obj)
    await resolver.load(obj)
    assert 0.1 < time.monotonic() - t0 < 0.15

    assert load_count == 1


@pytest.mark.asyncio
async def test_multi_resolve_concurrent_loads_once():
    output_manager = OutputManager(None, show_progress=False)
    resolver = Resolver(output_manager, client=None, environment_name="", app_id=None)

    load_count = 0

    class _DumbHandle(_Handle):
        pass

    class _DumbProvider(_Provider[_DumbHandle]):
        pass

    async def _load(resolver: Resolver, existing_object_id: Optional[str], handle: _DumbHandle):
        nonlocal load_count
        load_count += 1
        await asyncio.sleep(0.1)

    obj = _DumbProvider._from_loader(_load, "DumbProvider()")
    t0 = time.monotonic()
    await asyncio.gather(resolver.load(obj), resolver.load(obj))
    assert 0.1 < time.monotonic() - t0 < 0.15
    assert load_count == 1
