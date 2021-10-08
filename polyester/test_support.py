import asyncio

from . import Session

SLEEP_DELAY = 0.1

session = Session()


@session.function
def square(x):
    return x * x


@session.function
async def square_async(x):
    await asyncio.sleep(SLEEP_DELAY)
    return x * x


@session.function
def square_sync_returning_async(x):
    async def square():
        await asyncio.sleep(SLEEP_DELAY)
        return x * x

    return square()


@session.function
def raises(x):
    raise Exception("Failure!")


if __name__ == "__main__":
    raise Exception("This line is not supposed to be reachable")
