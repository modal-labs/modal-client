import asyncio
from . import base_image

SLEEP_DELAY = 0.1


@base_image.function
def square(x):
    return x*x


@base_image.function
async def square_async(x):
    await asyncio.sleep(SLEEP_DELAY)
    return x*x


@base_image.function
def square_sync_returning_async(x):
    async def square():
        await asyncio.sleep(SLEEP_DELAY)
        return x*x
    return square()


@base_image.function
def raises(x):
    raise Exception('Failure!')


if __name__ == '__main__':
    raise Exception('This line is not supposed to be reachable')
