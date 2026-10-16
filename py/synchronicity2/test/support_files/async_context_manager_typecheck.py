# Copyright Modal Labs 2026
"""Consumer typing checks for async_context_manager wrappers."""

import async_context_manager
from typing_extensions import assert_type

# Class dunders - sync
resource = async_context_manager.AsyncResource("test")
with resource as r:
    assert_type(r, async_context_manager.AsyncResource)

# Function returning context manager - sync
with async_context_manager.managed_value() as v:
    assert_type(v, async_context_manager.Connection)

# Method returning context manager - sync
svc = async_context_manager.ServiceWithContextMethod()
with svc.connect() as v2:
    assert_type(v2, async_context_manager.Connection)

with async_context_manager.ServiceWithFactoryContexts.connect_class() as v3:
    assert_type(v3, async_context_manager.Connection)

with async_context_manager.ServiceWithFactoryContexts.connect_static() as v4:
    assert_type(v4, async_context_manager.Connection)


# Async checks
async def async_check():
    async with async_context_manager.AsyncResource("test") as ar:
        assert_type(ar, async_context_manager.AsyncResource)

    async with async_context_manager.managed_value() as av:
        assert_type(av, async_context_manager.Connection)

    svc2 = async_context_manager.ServiceWithContextMethod()
    async with svc2.connect() as av2:
        assert_type(av2, async_context_manager.Connection)

    async with async_context_manager.ServiceWithFactoryContexts.connect_class() as av3:
        assert_type(av3, async_context_manager.Connection)

    async with async_context_manager.ServiceWithFactoryContexts.connect_static() as av4:
        assert_type(av4, async_context_manager.Connection)
