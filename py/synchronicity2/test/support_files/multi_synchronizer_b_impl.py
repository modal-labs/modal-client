# Copyright Modal Labs 2026
"""Second of two wrapper targets with distinct generated synchronizer modules."""

import asyncio
import threading

from synchronicity2 import Module

wrapper_module = Module("multi_sync.b")


@wrapper_module.wrap_function()
async def thread_and_loop_b() -> tuple[int, int]:
    return (threading.current_thread().ident or 0, id(asyncio.get_running_loop()))
