import pytest

from polyester.ctx_mgr_utils import CtxMgr


class TestCtxMgr(CtxMgr):
    def __init__(self, x):
        self.x = x
        self.state = "created"

    @classmethod
    async def _create(cls):
        return TestCtxMgr(1)

    async def _start(self):
        self.state = "started"

    async def _stop(self, hard):
        self.state = "stopped"


@pytest.mark.asyncio
async def test_ctx_mgr():
    # Use it as a context managr
    async with TestCtxMgr(42) as t_ctx_mgr:
        assert t_ctx_mgr.state == "started"
        assert t_ctx_mgr.x == 42

        # Inside the context manager, the .current() returns the only instance
        t_singleton = await TestCtxMgr.current()
        assert t_singleton.x == 42

    # Make sure the object got closed
    assert t_ctx_mgr.state == "stopped"

    # There is no current objects, so this will create one
    t_singleton = await TestCtxMgr.current()
    assert t_singleton.x == 1

    # If we use it as a context manager, it will create a second one
    async with TestCtxMgr(123) as t_ctx_mgr:
        assert t_ctx_mgr.state == "started"
        assert t_ctx_mgr.x == 123

        # There are two right now, so the singleton helper will fail
        with pytest.raises(Exception):
            tcm_current = await TestCtxMgr.current()
