# Copyright Modal Labs 2023
import asyncio
import pytest
import threading
from unittest import mock

from modal.serving import aio_serve_stub, serve_stub
from .supports.skip import skip_old_py, skip_windows


@pytest.mark.asyncio
async def test_live_reload(test_dir, server_url_env, servicer):
    stub_file = str(test_dir / "supports" / "app_run_tests" / "webhook.py")
    async with aio_serve_stub(stub_file):
        await asyncio.sleep(3.0)
    assert servicer.app_set_objects_count == 1
    assert servicer.app_client_disconnect_count == 1
    assert servicer.app_get_logs_initial_count == 1


@skip_old_py("live-reload requires python3.8 or higher", (3, 8))
@skip_windows("live-reload not supported on windows")
def test_file_changes_trigger_reloads(test_dir, server_url_env, servicer):
    watcher_done = threading.Event()

    async def fake_watch():
        for i in range(3):
            yield
        watcher_done.set()

    stub_file = str(test_dir / "supports" / "app_run_tests" / "webhook.py")
    with serve_stub(stub_file, _watcher=fake_watch()) as app:
        watcher_done.wait()  # wait until watcher loop is done

    assert servicer.app_set_objects_count == 4  # 1 + number of file changes
    assert servicer.app_client_disconnect_count == 1
    assert servicer.app_get_logs_initial_count == 1
    assert app.foo.web_url.startswith("http://")


@pytest.mark.asyncio
async def test_no_change(test_dir, server_url_env, servicer):
    async def fake_watch():
        # Iterator that returns immediately, yielding nothing
        if False:
            yield

    stub_file = str(test_dir / "supports" / "app_run_tests" / "webhook.py")
    async with aio_serve_stub(stub_file, _watcher=fake_watch()):
        pass

    assert servicer.app_set_objects_count == 1  # Should create the initial app once
    assert servicer.app_client_disconnect_count == 1
    assert servicer.app_get_logs_initial_count == 1


@pytest.mark.asyncio
async def test_heartbeats(test_dir, server_url_env, servicer):
    with mock.patch("modal.runner.HEARTBEAT_INTERVAL", 1):
        stub_file = str(test_dir / "supports" / "app_run_tests" / "webhook.py")
        async with aio_serve_stub(stub_file):
            await asyncio.sleep(3.1)

    apps = list(servicer.app_heartbeats.keys())
    assert len(apps) == 1
    assert servicer.app_heartbeats[apps[0]] == 4  # 0s, 1s, 2s, 3s
