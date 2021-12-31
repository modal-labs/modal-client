import asyncio

from . import Session, function

SLEEP_DELAY = 0.1

session = Session()


@function(session=session)
def square(x):
    return x * x


@function(session=session)
async def square_async(x):
    await asyncio.sleep(SLEEP_DELAY)
    return x * x


@function(session=session)
def square_sync_returning_async(x):
    async def square():
        await asyncio.sleep(SLEEP_DELAY)
        return x * x

    return square()


@function(session=session)
def raises(x):
    raise Exception("Failure!")


if __name__ == "__main__":
    raise Exception("This line is not supposed to be reachable")
