# Copyright Modal Labs 2025
import asyncio
from collections.abc import Callable


class IdleCountdown:
    """An extendable idle timeout tracker."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._handle: asyncio.TimerHandle | None = None
        self._retired = False
        self._expire_on_idle = False
        self._release: Callable[[], None] | None = None

    def _assert_owning_loop(self) -> None:
        assert asyncio.get_running_loop() is self._loop, "Idle bookkeeping must run on its owning event loop"

    def expire_when_idle(self) -> None:
        """Release at the current or next idle boundary, even if the timeout is disabled.

        Must run on the owning event loop without yielding control.
        """
        self._assert_owning_loop()
        self._expire_on_idle = True
        if self._release is not None and not self._retired:
            release = self._release
            self.stop()
            release()

    def stop(self) -> None:
        """Cancel the countdown on the owning event loop without yielding control."""
        self._assert_owning_loop()
        self._release = None
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def retire(self) -> None:
        """Stop the countdown and refuse every subsequent arm.

        Must run on the owning event loop without yielding control.
        """
        self._assert_owning_loop()
        self._retired = True
        self.stop()

    def arm(self, timeout_secs: float, release: Callable[[], None]) -> None:
        """Start the countdown again, calling `release` when it elapses.

        Must run on the owning event loop without yielding control.
        A timeout of zero or less never fires unless `expire_when_idle` was
        requested. `release` runs as a plain callback on the event loop, so it
        must not block.
        """
        self.stop()
        if self._retired:
            return
        self._release = release
        if self._expire_on_idle:
            timeout_secs = 0
        elif timeout_secs <= 0:
            return
        self._handle = self._loop.call_later(timeout_secs, release)
