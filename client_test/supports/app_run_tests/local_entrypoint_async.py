# Copyright Modal Labs 2022

import modal

stub = modal.Stub()


@stub.function()
def foo():
    pass


@stub.local_entrypoint()
async def main():
    print("called locally (async)")
    await foo.remote.aio()
    await foo.remote.aio()
