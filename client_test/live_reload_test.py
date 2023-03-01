# Copyright Modal Labs 2023
import asyncio
import pytest
from unittest import mock

from modal._live_reload import aio_run_serve_loop, run_serve_loop
from .supports.skip import skip_old_py, skip_windows


@pytest.mark.asyncio
async def test_live_reload(test_dir, server_url_env, servicer):
    stub_file = str(test_dir / "supports" / "app_run_tests" / "webhook.py")
    await aio_run_serve_loop(stub_file, timeout=3.0)
    assert servicer.app_set_objects_count == 1
    assert servicer.app_client_disconnect_count == 1
    assert servicer.app_get_logs_initial_count == 1


@skip_old_py("live-reload requires python3.8 or higher", (3, 8))
@skip_windows("live-reload not supported on windows")
def test_file_changes_trigger_reloads(test_dir, server_url_env, servicer):
    async def fake_watch():
        for i in range(3):
            yield

    app_q: asyncio.Queue = asyncio.Queue()
    stub_file = str(test_dir / "supports" / "app_run_tests" / "webhook.py")
    run_serve_loop(stub_file, _watcher=fake_watch(), _app_q=app_q)
    assert servicer.app_set_objects_count == 4  # 1 + number of file changes
    assert servicer.app_client_disconnect_count == 1
    assert servicer.app_get_logs_initial_count == 1
    app = app_q.get_nowait()
    assert app.foo.web_url.startswith("http://")


@pytest.mark.asyncio
async def test_no_change(test_dir, server_url_env, servicer):
    async def fake_watch():
        # Iterator that returns immediately, yielding nothing
        if False:
            yield

    stub_file = str(test_dir / "supports" / "app_run_tests" / "webhook.py")
    await aio_run_serve_loop(stub_file, _watcher=fake_watch())
    assert servicer.app_set_objects_count == 1  # Should create the initial app once
    assert servicer.app_client_disconnect_count == 1
    assert servicer.app_get_logs_initial_count == 1


@pytest.mark.asyncio
async def test_heartbeats(test_dir, server_url_env, servicer):
    with mock.patch("modal.stub.HEARTBEAT_INTERVAL", 1):
        stub_file = str(test_dir / "supports" / "app_run_tests" / "webhook.py")
        await aio_run_serve_loop(stub_file, timeout=3.5)

    apps = list(servicer.app_heartbeats.keys())
    assert len(apps) == 1
    assert servicer.app_heartbeats[apps[0]] == 4  # 0s, 1s, 2s, 3s
