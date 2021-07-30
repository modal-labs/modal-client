import asyncio
import inspect
import sys
import traceback

from .async_utils import retry
from .client import ContainerClient
from .config import logger
from .function import Function
from .proto import api_pb2


async def function(client, task_id, function_id, target):
    # Note that there's actually 4 types of functions in Python: (is generator) * (is async)
    # TODO: the function could be a vanilla function returning a generator or a coroutine, so
    # we should really just call it and see what we get back (the only caveat is that it has
    # to be called on a separate thread since it *might* be a blocking function.
    assert callable(target)
    is_generator = inspect.isgeneratorfunction(target) or inspect.isasyncgenfunction(target)
    is_async = inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target)
    logger.debug('Function is generator: %s is async: %s' % (is_generator, is_async))

    while True:
        data, input_id, stop = await client.function_get_next_input(task_id, function_id)
        if stop:
            logger.debug('Function calls exhausted, exiting')
            break
        args, kwargs = data

        # Values to send to the server
        fv_data, fv_status, fv_exception, fv_traceback = None, None, None, None
        try:
            if asyncio.iscoroutinefunction(target):
                res = await target(*args, **kwargs)
            else:
                target_bound = lambda: target(*args, **kwargs)
                res = await asyncio.get_running_loop().run_in_executor(None, target_bound)
            fv_data = res
            fv_status = api_pb2.GenericResult.Status.SUCCESS
        except Exception as exc:
            # Note that we have to stringify the exception since a Python object isn't always possible to unpickle on the client side
            fv_exception = repr(exc)
            fv_traceback = traceback.format_exc()
            fv_status = api_pb2.GenericResult.Status.FAILURE

        await client.function_output(input_id, fv_status, fv_data, fv_exception, fv_traceback)


async def main(args):
    tag, task_id, function_id, module_name, function_name = args
    assert tag == 'function'
    logger.debug('Container: starting')

    logger.debug('Getting function %s.%s' % (module_name, function_name))
    target = Function.get_function(module_name, function_name)

    async with ContainerClient(task_id) as client:
        await function(client, task_id, function_id, target)

    logger.debug('Container: done')


if __name__ == '__main__':
    args = sys.argv[1:]
    asyncio.run(main(args))
