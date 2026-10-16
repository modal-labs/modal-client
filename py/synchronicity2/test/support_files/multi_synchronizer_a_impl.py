# Copyright Modal Labs 2026
"""First of two wrapper targets with distinct generated synchronizer modules."""

import asyncio
import threading

from synchronicity2 import Module

wrapper_module = Module("multi_sync.a")


@wrapper_module.wrap_function()
async def thread_and_loop_a() -> tuple[int, int]:
    """Return (thread id, event loop id) from this synchronizer's worker thread."""
    return (threading.current_thread().ident or 0, id(asyncio.get_running_loop()))
