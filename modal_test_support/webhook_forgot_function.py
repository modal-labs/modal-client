# Copyright Modal Labs 2023
from modal import web_endpoint


@web_endpoint()
async def absent_minded_function(x):
    pass
