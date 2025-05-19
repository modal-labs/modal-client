import asyncio
import time
from contextlib import asynccontextmanager

from modal.config import config, logger


@asynccontextmanager
async def _event_loop_monitor(monitor_period: float, warning_threshold: float):
    async def monitor():
        while 1:
            t0 = time.monotonic()
            await asyncio.sleep(monitor_period)
            duration = time.monotonic() - t0
            delay = duration - monitor_period
            if delay >= warning_threshold:
                logger.warning(
                    f"Detected an event loop delay of {delay:.2f}s.\n"
                    f"This indicates there is something non-async running in an async method for a "
                    f"significant amount of time, which prevents functioning concurrency."
                )
                # TODO: add hints for user to find the problem?

    loop_task = asyncio.create_task(monitor())
    try:
        yield
    finally:
        loop_task.cancel()


@asynccontextmanager
async def configurable_event_loop_monitor():
    if not config.get("event_loop_monitor"):
        # if user disabled event loop monitor
        yield
        return

    monitor_period = config.get("event_loop_monitor_period_ms")
    warning_threshold = config.get("event_loop_monitor_threshold_ms")
    async with _event_loop_monitor(monitor_period, warning_threshold):
        yield
