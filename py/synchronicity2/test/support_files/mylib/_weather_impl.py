# Copyright Modal Labs 2026
"""Weather implementation module for README-style integration and markdown tests.

Imports ``Module`` from the vendored runtime package, matching typical library layout.
"""

from __future__ import annotations

import collections.abc

from mylib.synchronicity2 import Module

wrapper_module = Module("mylib.weather")


@wrapper_module.wrap_function()
async def get_temperature(city: str) -> float:
    return 20.0 if city == "Stockholm" else 0.0


@wrapper_module.wrap_class()
class WeatherClient:
    default_city: str

    def __init__(self, default_city: str):
        self.default_city = default_city

    async def current(self) -> float:
        return 21.0


@wrapper_module.wrap_function()
async def stream_temperature_readings() -> collections.abc.AsyncGenerator[float, None]:
    """Async generator of sample readings (°C)."""
    yield 17.5
    yield 18.0
    yield 18.5
