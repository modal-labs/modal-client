import asyncio

from . import App, function

SLEEP_DELAY = 0.1

app = App()


@function(app=app)
def square(x):
    return x * x


@function(app=app)
async def square_async(x):
    await asyncio.sleep(SLEEP_DELAY)
    return x * x


@function(app=app)
def square_sync_returning_async(x):
    async def square():
        await asyncio.sleep(SLEEP_DELAY)
        return x * x

    return square()


@function(app=app)
def raises(x):
    raise Exception("Failure!")


if __name__ == "__main__":
    raise Exception("This line is not supposed to be reachable")
