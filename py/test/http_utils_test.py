# Copyright Modal Labs 2026
import asyncio
import gc
import pytest
import weakref

from modal._utils.async_utils import _shutdown_tasks
from modal._utils.http_utils import ClientSessionRegistry


@pytest.mark.parametrize("explicit_close", [False, True])
def test_session_registry_releases_loop(explicit_close, monkeypatch):
    monkeypatch.setattr(ClientSessionRegistry, "_client_session_active", False)
    if hasattr(ClientSessionRegistry, "_client_session"):
        monkeypatch.setattr(ClientSessionRegistry, "_client_session", None)
    monkeypatch.setattr(ClientSessionRegistry, "_shutdown_unregister", None)

    async def cycle():
        session = ClientSessionRegistry.get_session()
        assert ClientSessionRegistry.get_session() is session
        refs = [weakref.ref(session), weakref.ref(asyncio.get_running_loop())]
        await asyncio.sleep(0)
        if explicit_close:
            await ClientSessionRegistry.close_session()
            await ClientSessionRegistry.close_session()
            assert session.closed
        return refs

    count = len(_shutdown_tasks)
    for _ in range(3):
        refs = asyncio.run(cycle())
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert not ClientSessionRegistry._client_session_active
        assert len(_shutdown_tasks) <= count
