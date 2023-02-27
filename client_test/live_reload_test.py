# Copyright Modal Labs 2023
import pytest

from modal._live_reload import aio_run_serve_loop, run_serve_loop
from .supports.skip import skip_old_py, skip_windows


@pytest.mark.asyncio
async def test_live_reload(test_dir, server_url_env, servicer):
    stub_file = str(test_dir / "supports" / "app_run_tests" / "default_stub.py")
    await aio_run_serve_loop(stub_file, timeout=3.0)
    assert servicer.app_set_objects_count == 1


@skip_old_py("live-reload requires python3.8 or higher", (3, 8))
@skip_windows("live-reload not supported on windows")
def test_file_changes_trigger_reloads(test_dir, server_url_env, servicer):
    async def fake_watch():
        yield  # dummy at the beginning
        for i in range(3):
            yield

    stub_file = str(test_dir / "supports" / "app_run_tests" / "default_stub.py")
    run_serve_loop(stub_file, _watcher=fake_watch())
    assert servicer.app_set_objects_count == 4  # 1 + number of file changes


@skip_old_py("live-reload requires python3.8 or higher", (3, 8))
@skip_windows("live-reload not supported on windows")
@pytest.mark.asyncio
async def test_no_change(test_dir, server_url_env, servicer):
    async def fake_watch():
        # Iterator that never yields
        if False:
            yield

    stub_file = str(test_dir / "supports" / "app_run_tests" / "default_stub.py")
    await aio_run_serve_loop(stub_file, _watcher=fake_watch())
    assert servicer.app_set_objects_count == 0
