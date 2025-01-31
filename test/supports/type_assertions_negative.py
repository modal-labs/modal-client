# Copyright Modal Labs 2025
import modal

app = modal.App()


@app.function()
def typed_func(a: str) -> float:
    return 0.0


typed_func.remote(b="hello")  # wrong arg name
typed_func.remote(a=10)  # wrong arg type

typed_func.local(c="hello")  # wrong arg name
typed_func.local(a=10)  # wrong arg type


async def aio_calls() -> None:
    await typed_func.remote.aio(e="hello")  # wrong arg name
    await typed_func.remote.aio(a=10.5)  # wrong arg type
