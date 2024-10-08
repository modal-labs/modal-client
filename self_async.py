import asyncio
import random
from contextlib import asynccontextmanager


async def async_generator(count, multiplier=1):
    for i in range(count):
        # sleep = random.random()
        # print(f"async_generator: {i=} {sleep=}\n")
        await asyncio.sleep(0.1)

        res = (i + 1) * multiplier
        yield res


async def async_generator_error(count, multiplier=1):
    for i in range(count):
        # sleep = random.random()
        # print(f"async_generator: {i=} {sleep=}\n")
        await asyncio.sleep(0.1)

        res = (i + 1) * multiplier
        if res == 30:
            print("raising exception")
            raise Exception("res == 30")
        yield res


def func(x):
    return f"func({x})"


async def async_func(x):
    await asyncio.sleep(random.random())
    return f"async_func({x})"


async def async_func_error(x):
    if x == 5:
        raise Exception("x == 5")
    await asyncio.sleep(random.random())
    return f"async_func({x})"


@asynccontextmanager
async def aclosing(agen):
    try:
        yield agen
    finally:
        await agen.aclose()


async def async_map(input, callable, concurrency=1):
    input_queue = asyncio.Queue(maxsize=concurrency)
    results_queue = asyncio.Queue()

    new_result_event = asyncio.Event()

    async def producer():
        async for item in input:
            await input_queue.put(item)

        # as long as there are inputs
        #

    async def worker():
        while True:
            try:
                item = await input_queue.get()

                # check if callable is async
                if asyncio.iscoroutinefunction(callable):
                    result = await callable(item)
                else:
                    result = callable(item)

                # result = await callable(item)
                await results_queue.put(result)
                new_result_event.set()
                # input_queue.task_done()
            except Exception as e:
                await results_queue.put(e)
                new_result_event.set()
            finally:
                input_queue.task_done()

    producer_task = asyncio.create_task(producer())
    worker_tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]

    wait_for_results_task = asyncio.create_task(new_result_event.wait())

    async def complete_map():
        await producer_task
        await input_queue.join()

    complete_map_task = asyncio.create_task(complete_map())

    try:
        while True:
            await asyncio.wait(
                [complete_map_task, producer_task, *worker_tasks, wait_for_results_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if complete_map_task.done():
                while not results_queue.empty():
                    yield await results_queue.get()
                break

            if new_result_event.is_set():
                while not results_queue.empty():
                    result = await results_queue.get()
                    if isinstance(result, Exception):
                        raise result
                    yield result
                new_result_event.clear()

    finally:
        for task in [producer_task, complete_map_task, *worker_tasks]:
            task.cancel()
        await asyncio.gather(producer_task, complete_map_task, *worker_tasks, return_exceptions=True)


async def async_merge(input, *more_inputs):
    queue = asyncio.Queue()
    inputs = [input] + list(more_inputs)

    async def producer(iterator):
        async for item in iterator:
            await queue.put(item)

    tasks = [asyncio.create_task(producer(it)) for it in inputs]

    async def complete_merge():
        for task in tasks:
            await task
        await queue.join()

    complete_merge_task = asyncio.create_task(complete_merge())

    try:
        while True:
            await asyncio.wait([complete_merge_task, *tasks], return_when=asyncio.FIRST_COMPLETED)
            if complete_merge_task.done():
                break

            while not queue.empty():
                item = await queue.get()
                # if isinstance(item, Exception):
                #     print("raising exception when getting item")
                #     raise item
                yield item
                queue.task_done()

    finally:
        for task in [complete_merge_task, *tasks]:
            task.cancel()
        await asyncio.gather(complete_merge_task, *tasks, return_exceptions=False)


async def main():
    # start = time.time()
    # res = [i async for i in async_map(async_generator(10), async_func, concurrency=1)]
    # end = time.time()
    # print(f"time: {end - start}\n")

    async with aclosing(
        async_merge(async_generator(5, 1), async_generator_error(5, 10), async_generator(5, 100))
    ) as stream:
        async for i in stream:
            print(i)

    # import time
    # start = time.time()
    # res = []
    # async with aclosing(async_map(async_generator(10), async_func_error, concurrency=10)) as stream:
    #     async for i in stream:
    #         res.append(i)

    # # import aiostream
    # # async with aiostream.stream.map(async_generator(10), async_func_error, task_limit=10).stream() as stream:
    # #     async for i in stream:
    # #         res.append(i)
    # end = time.time()
    # print(f"time: {end - start}\n")

    # for r in res:
    #     print(r)


if __name__ == "__main__":
    # print("Hello World")

    asyncio.run(main())
