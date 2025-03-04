# Copyright Modal Labs 2023
from modal import fastapi_endpoint


@fastapi_endpoint()
async def absent_minded_function(x):
    pass
