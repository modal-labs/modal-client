# Copyright Modal Labs 2022
from __future__ import annotations

import modal.experimental
from modal import (
    App,
    enter,
    method,
)

app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@app.cls()
class StopFetching:
    @enter()
    def init(self):
        self.counter = 0

    @method()
    def after_two(self, x):
        self.counter += 1

        if self.counter >= 2:
            modal.experimental.stop_fetching_inputs()

        return x * x
