# Copyright Modal Labs 2023
import asyncio
import pytest
import threading
from unittest import mock

from modal import Function
from modal.serving import serve_stub

from .supports.app_run_tests.webhook import stub
from .supports.skip import skip_old_py, skip_windows


@pytest.fixture
def stub_ref(test_dir):
    return str(test_dir / "supports" / "app_run_tests" / "webhook.py")


@pytest.mark.asyncio
async def test_live_reload(stub_ref, server_url_env, servicer):
    async with serve_stub.aio(stub, stub_ref):
        await asyncio.sleep(3.0)
    assert servicer.app_set_objects_count == 1
    assert servicer.app_client_disconnect_count == 1
    assert servicer.app_get_logs_initial_count == 1


@skip_old_py("live-reload requires python3.8 or higher", (3, 8))
@skip_windows("live-reload not supported on windows")
def test_file_changes_trigger_reloads(stub_ref, server_url_env, servicer):
    watcher_done = threading.Event()

    async def fake_watch():
        for i in range(3):
            yield
        watcher_done.set()

    with serve_stub(stub, stub_ref, _watcher=fake_watch()):
        watcher_done.wait()  # wait until watcher loop is done

    assert servicer.app_set_objects_count == 4  # 1 + number of file changes
    assert servicer.app_client_disconnect_count == 1
    assert servicer.app_get_logs_initial_count == 1
    assert isinstance(stub.foo, Function)
    assert stub.foo.web_url.startswith("http://")


@pytest.mark.asyncio
async def test_no_change(stub_ref, server_url_env, servicer):
    async def fake_watch():
        # Iterator that returns immediately, yielding nothing
        if False:
            yield

    async with serve_stub.aio(stub, stub_ref, _watcher=fake_watch()):
        pass

    assert servicer.app_set_objects_count == 1  # Should create the initial app once
    assert servicer.app_client_disconnect_count == 1
    assert servicer.app_get_logs_initial_count == 1


@pytest.mark.asyncio
@skip_windows("this flakes a lot of the time by giving 5 heartbeats")
async def test_heartbeats(stub_ref, server_url_env, servicer):
    with mock.patch("modal.runner.HEARTBEAT_INTERVAL", 1):
        async with serve_stub.aio(stub, stub_ref):
            await asyncio.sleep(3.1)

    apps = list(servicer.app_heartbeats.keys())
    assert len(apps) == 1
    assert servicer.app_heartbeats[apps[0]] == 4  # 0s, 1s, 2s, 3s
