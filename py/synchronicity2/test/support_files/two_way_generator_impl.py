# Copyright Modal Labs 2026
"""Test module for two-way generators (generators that use send())."""

from typing import AsyncGenerator, Generator, Iterator

from synchronicity2 import Module

# Create the wrapper module
wrapper_module = Module("two_way_generator")


@wrapper_module.wrap_class()
class Payload:
    value: str

    def __init__(self, value: str):
        self.value = value


@wrapper_module.wrap_function()
async def echo_generator() -> AsyncGenerator[str, str]:
    """A two-way generator that echoes back what you send to it.

    This generator demonstrates the send() functionality:
    - First yield returns a greeting
    - Subsequent yields echo back what was sent
    - Transforms the sent value by adding a prefix
    """
    sent_value = yield "Ready"

    while True:
        sent_value = yield f"Echo: {sent_value}"


@wrapper_module.wrap_function()
async def accumulator_generator() -> AsyncGenerator[int, int]:
    """A two-way generator that accumulates sent values.

    This generator maintains state and returns the running sum:
    - First yield returns 0 (initial sum)
    - Each send() adds to the sum and yields the new total
    """
    total = 0
    sent_value = yield total

    while True:
        total += sent_value
        sent_value = yield total


@wrapper_module.wrap_function()
async def multiplier_generator(factor: int) -> AsyncGenerator[int, int]:
    """A two-way generator that multiplies sent values by a factor.

    Args:
        factor: The multiplication factor

    Yields:
        The product of sent value and factor
    """
    sent_value = yield 0  # Initial value

    while True:
        sent_value = yield sent_value * factor


@wrapper_module.wrap_function()
async def async_payload_generator() -> AsyncGenerator[Payload, Payload]:
    sent_value = yield Payload("ready")
    while True:
        sent_value = yield sent_value


@wrapper_module.wrap_function()
def sync_payload_generator() -> Generator[Payload, Payload, Payload]:
    sent_value = yield Payload("ready")
    return sent_value


@wrapper_module.wrap_function()
def iter_payloads() -> Iterator[Payload]:
    yield Payload("first")
    yield Payload("second")


# Global list to track cleanup calls for testing
cleanup_tracker = []


@wrapper_module.wrap_function()
async def generator_with_cleanup() -> AsyncGenerator[str, None]:
    """A generator that tracks cleanup via aclose().

    This generator demonstrates proper cleanup behavior:
    - Yields several values
    - Records when it's being closed via finally block
    - Includes actual async operation in cleanup to test true async cleanup
    - Uses 1 second sleep to verify cleanup is properly awaited
    - Allows testing that aclose() is properly forwarded
    """
    import asyncio

    cleanup_tracker.clear()  # Reset tracker
    try:
        yield "first"
        yield "second"
        yield "third"
    finally:
        # This should run when aclose() is called
        # Include actual async operation to test true async cleanup
        # Use 1 second to verify the cleanup is actually awaited
        await asyncio.sleep(1.0)
        cleanup_tracker.append("cleanup_called")
