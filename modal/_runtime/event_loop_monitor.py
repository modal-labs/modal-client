import asyncio
import time
from contextlib import asynccontextmanager
from typing import Callable

from modal.config import config, logger


@asynccontextmanager
async def event_loop_monitor(
    monitor_period_seconds: float, warning_threshold_seconds: float, callback: Callable[[float], None]
):
    started = asyncio.Event()
    stop_event = asyncio.Event()

    async def monitor():
        while not stop_event.is_set():
            t0 = time.monotonic()
            started.set()
            done, _ = await asyncio.wait([asyncio.create_task(stop_event.wait())], timeout=monitor_period_seconds)
            duration = time.monotonic() - t0
            delay = duration - monitor_period_seconds
            if delay >= warning_threshold_seconds:
                callback(delay)

            if done:
                # stop event triggered (could be after the set timeout though!)
                break

    loop_task = asyncio.create_task(monitor())
    await started.wait()
    try:
        yield
    finally:
        stop_event.set()
        await loop_task


@asynccontextmanager
async def configurable_event_loop_monitor():
    if not config.get("event_loop_monitor"):
        # if user disabled event loop monitor
        yield
        return

    def callback(delay: float) -> None:
        # TODO: add hints for user to find the problem?
        logger.warning(
            f"Detected an asyncio event loop delay of {delay:.2f}s.\n"
            f"This indicates something non-async is running in an async method for a "
            f"which could prevent functioning concurrency.\n"
            f"If this is intentional, you can deactivate this warning with MODAL_EVENT_LOOP_MONITOR=0"
        )

    monitor_period = config.get("event_loop_monitor_period_ms") / 1000.0
    warning_threshold = config.get("event_loop_monitor_threshold_ms") / 1000.0
    async with event_loop_monitor(monitor_period, warning_threshold, callback):
        yield
