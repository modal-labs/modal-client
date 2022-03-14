import asyncio

from . import App

SLEEP_DELAY = 0.1

app = App()


@app.function()
def square(x):
    return x * x


@app.function()
async def square_async(x):
    await asyncio.sleep(SLEEP_DELAY)
    return x * x


@app.function()
def square_sync_returning_async(x):
    async def square():
        await asyncio.sleep(SLEEP_DELAY)
        return x * x

    return square()


@app.function()
def raises(x):
    raise Exception("Failure!")


if __name__ == "__main__":
    raise Exception("This line is not supposed to be reachable")
