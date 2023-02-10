# Copyright Modal Labs 2023
import asyncio
import os
import sys
import pytest

try:
    from unittest.mock import AsyncMock
except ImportError:
    # Support Python 3.7
    from unittest.mock import MagicMock

    class AsyncMock(MagicMock):  # type: ignore
        async def __call__(self, *args, **kwargs):
            return super(AsyncMock, self).__call__(*args, **kwargs)  # type: ignore


from modal import Stub
from modal._live_reload import MODAL_AUTORELOAD_ENV
from modal.aio import AioStub


def dummy():
    pass


class FakeProcess:
    def send_signal(self, signal):
        pass

    def terminate(self):
        pass


@pytest.mark.skipif(sys.version_info < (3, 8), reason="live-reload requires python3.8 or higher")
def test_file_changes_trigger_reloads(client, monkeypatch, servicer, test_dir):
    async def fake_watch(mounts, output_mgr, timeout):
        for i in range(3):
            yield

    stub = Stub()
    stub.webhook(dummy)

    mock_create_subprocess_exec = AsyncMock(return_value=FakeProcess())
    monkeypatch.setattr("modal.stub.asyncio.create_subprocess_exec", mock_create_subprocess_exec)
    monkeypatch.setattr("modal._watcher.watch", fake_watch)

    stub.serve(client=client, timeout=None)
    assert mock_create_subprocess_exec.call_count == 3


@pytest.mark.asyncio
async def test_reloadable_serve_ignores_file_changes(client, monkeypatch, servicer, test_dir):
    async def fake_watch(stub, output_mgr, timeout):
        # Iterator that never yields
        if False:
            yield

    stub = AioStub()
    stub.webhook(dummy)

    mock_create_subprocess_exec = AsyncMock(return_value=FakeProcess())
    monkeypatch.setattr("modal.stub.asyncio.create_subprocess_exec", mock_create_subprocess_exec)
    monkeypatch.setattr("modal._watcher.watch", fake_watch)

    # The app should not react to AppChange.TIMEOUT, and instead need
    # the wait_for to cancel it.
    monkeypatch.setattr(os, "environ", {MODAL_AUTORELOAD_ENV: "ap-12345"})
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stub.serve(client=client), timeout=1.0)

    assert mock_create_subprocess_exec.call_count == 0
