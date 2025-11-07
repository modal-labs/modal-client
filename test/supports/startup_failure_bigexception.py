# Copyright Modal Labs 2022
import modal

app = modal.App("hello-world", include_source=False)


class BigException(Exception):
    def __init__(self, message):
        self.full_message = message
        self.message = message[:10]
        super().__init__(self.message)


if not modal.is_local():
    raise BigException("a" * 5_000_000)


@app.function()
def f(i):
    pass
