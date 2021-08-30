import asyncio
import inspect
import sys
import traceback
import uuid

from .async_utils import retry, synchronizer
from .client import Client
from .config import logger
from .grpc_utils import GRPC_REQUEST_TIMEOUT, BLOCKING_REQUEST_TIMEOUT
from .function import Function
from .proto import api_pb2


@synchronizer
class FunctionRunner:
    def __init__(self, client, task_id, function_id, target):
        self.client = client
        self.task_id = task_id
        self.function_id = function_id
        self.target = target

        # Note that there's actually 4 types of functions in Python: (is generator) * (is async)
        # TODO: the function could be a vanilla function returning a generator or a coroutine, so
        # we should really just call it and see what we get back (the only caveat is that it has
        # to be called on a separate thread since it *might* be a blocking function.
        assert callable(target)
        self.is_generator = inspect.isgeneratorfunction(target) or inspect.isasyncgenfunction(target)
        self.is_async = inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target)
        logger.debug('Function is generator: %s is async: %s' % (self.is_generator, self.is_async))

    async def run(self):
        while True:
            data, input_id, stop = await self._get_next_input()
            if stop:
                logger.debug('Function calls exhausted, exiting')
                break
            args, kwargs = data

            # Values to send to the server
            fv_data, fv_status, fv_exception, fv_traceback = None, None, None, None
            try:
                if asyncio.iscoroutinefunction(self.target):
                    res = await target(*args, **kwargs)
                else:
                    target_bound = lambda: self.target(*args, **kwargs)
                    res = await asyncio.get_running_loop().run_in_executor(None, target_bound)
                fv_data = res
                fv_status = api_pb2.GenericResult.Status.SUCCESS
            except Exception as exc:
                # Note that we have to stringify the exception since a Python object isn't always possible to unpickle on the client side
                fv_exception = repr(exc)
                fv_traceback = traceback.format_exc()
                fv_status = api_pb2.GenericResult.Status.FAILURE

            await self._output(input_id, fv_status, fv_data, fv_exception, fv_traceback)

    async def _get_next_input(self):
        while True:
            idempotency_key = str(uuid.uuid4())
            request = api_pb2.FunctionGetNextInputRequest(
                task_id=self.task_id,
                function_id=self.function_id,
                idempotency_key=idempotency_key,
                timeout=BLOCKING_REQUEST_TIMEOUT
            )
            response = await retry(self.client.stub.FunctionGetNextInput)(request, timeout=GRPC_REQUEST_TIMEOUT)
            if response.stop:
                return (None, None, True)
            if response.input_id:
                break
        return (self.client.deserialize(response.data), response.input_id, False)

    async def _output(self, input_id, status, data, exception: str, traceback: str):
        data_serialized = self.client.serialize(data)
        output = api_pb2.GenericResult(
            status=status,
            data=data_serialized,
            exception=exception,
            traceback=traceback
        )
        idempotency_key = str(uuid.uuid4())
        request = api_pb2.FunctionOutputRequest(input_id=input_id, idempotency_key=idempotency_key, output=output)
        await retry(self.client.stub.FunctionOutput)(request)


async def main(args):
    tag, task_id, function_id, module_name, function_name = args
    assert tag == 'function'
    logger.debug('Container: starting')

    logger.debug('Getting function %s.%s' % (module_name, function_name))
    target = Function.get_function(module_name, function_name)

    client = await Client.from_env()
    function_runner = FunctionRunner(client, task_id, function_id, target)
    await function_runner.run()
    logger.debug('Container: done')


if __name__ == '__main__':
    args = sys.argv[1:]
    asyncio.run(main(args))
