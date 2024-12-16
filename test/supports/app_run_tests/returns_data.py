# Copyright Modal Labs 2024
import modal

app = modal.App()


@app.local_entrypoint()
def returns_str() -> str:
    return "Hello!"


@app.local_entrypoint()
def returns_bytes() -> bytes:
    return b"Hello!"


@app.local_entrypoint()
def returns_int() -> int:
    return 42
