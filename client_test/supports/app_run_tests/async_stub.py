# Copyright Modal Labs 2022
import modal.aio

stub = modal.aio.AioStub()


@stub.function
async def foo():
    pass
