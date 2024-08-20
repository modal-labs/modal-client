# Copyright Modal Labs 2022
from __future__ import annotations

import modal.experimental
from modal import (
    App,
    enter,
    method,
)

app = App()


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


@app.cls(allow_concurrent_inputs=1, concurrency_limit=1)
class SetLocalConcurrentInputs:
    @enter()
    def init(self):
        modal.experimental.set_local_concurrent_inputs(20)

    @method()
    def get_concurrent_inputs(self):
        return modal.experimental.get_local_concurrent_inputs()
