import asyncio
import cloudpickle
import importlib
import inspect
import os
import uuid

from .async_utils import retry, synchronizer
from .client import Client
from .config import logger
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .proto import api_pb2
from .queue import Queue
from .serialization import serializable


def _function_to_path(f):
    module = inspect.getmodule(f)
    if module.__spec__:
        module_name = module.__spec__.name
    else:
        fn = os.path.splitext(module.__file__)[0]
        path = os.path.relpath(fn, os.getcwd())  # TODO: might want to do it relative to other paths in sys.path?
        parts = os.path.split(path)
        module_name = '.'.join(p for p in parts if p and not p.startswith('.'))
    function_name = f.__name__
    return (module_name, function_name)


def _path_to_function(module_name, function_name):
    # Opposite of _function_to_path
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


# @serializable
@synchronizer
class Function:
    def __init__(self, raw_f, image=None, client=None):
        assert callable(raw_f)
        self.raw_f = raw_f
        self.module_name, self.function_name = _function_to_path(raw_f)
        self.name = '%s.%s' % (self.module_name, self.function_name)
        self.image = image
        self.client = client  # TODO: is this ever used?
        self.function_id = None

    def _get_client(self):
        # Maybe this needs to go on the serializable base class, not sure
        if self.client is None:
            return Client.get_default()
        else:
            return self.client

    async def _enqueue(self, client, call_id, args, star, kwargs):
        if not star:
            # Everything will just be passed as the first input
            args = [(arg,) for arg in args]
        request = api_pb2.FunctionCallRequest(
            function_id=self.function_id,
            inputs=[client.serialize((arg, kwargs)) for arg in args],
            idempotency_key=str(uuid.uuid4()),
            call_id=call_id
        )
        response = await retry(client.stub.FunctionCall)(request)
        return response.call_id

    async def _dequeue(self, client, call_id, n_outputs):
        while True:
            request = api_pb2.FunctionGetNextOutputRequest(
                function_id=self.function_id,
                call_id=call_id,
                timeout=BLOCKING_REQUEST_TIMEOUT,
                idempotency_key=str(uuid.uuid4()),
                n_outputs=n_outputs,
            )
            response = await retry(client.stub.FunctionGetNextOutput)(request, timeout=GRPC_REQUEST_TIMEOUT)
            if response.outputs:
                break
        for output in response.outputs:
            if output.status != api_pb2.GenericResult.Status.SUCCESS:
                raise Exception('Remote exception: %s\n%s' % (output.exception, output.traceback))
            yield client.deserialize(output.data)

    async def map(self, inputs, unordered=False, star=False, return_exceptions=False, window=100, kwargs={}):
        # Most of the complexity of this function comes from the input throttling.
        # Basically the idea is that we maintain x (default 100) outstanding requests at any point in time,
        # and we don't enqueue more requests until we get back enough values.
        # It probably makes a lot of sense to move the input throttling to the server instead.

        async with self._get_client() as client:
            if not self.function_id:
                # Create function remotely
                # First let's verify we got the path right (just make sure it doesn't raise)
                _path_to_function(self.module_name, self.function_name)

                image_id = await self.image.join(client)
                data = client.serialize(self.raw_f)
                request = api_pb2.FunctionGetOrCreateRequest(
                    client_id=client.client_id,
                    data=data,  # TODO: remove
                    image_id=image_id,
                    name=self.name,  # TODO: remove
                    image_local_id=self.image.local_id,
                    module_name=self.module_name,
                    function_name=self.function_name,
                )
                response = await client.stub.FunctionGetOrCreate(request)
                self.function_id = response.function_id

            # TODO: we should support asynchronous generators as well
            inputs = iter(inputs)  # Handle non-generator inputs
            call_id = None

            n_enqueued, n_dequeued = 0, 0
            input_exhausted = False
            while not input_exhausted or n_dequeued < n_enqueued:
                logger.debug('Map status: %d enqueued, %d dequeued' % (n_enqueued, n_dequeued))
                batch_args = []
                while not input_exhausted and n_enqueued < n_dequeued + window:
                    try:
                        batch_args.append(next(inputs))
                        n_enqueued += 1
                    except StopIteration:
                        input_exhausted = True
                if batch_args:
                    call_id = await self._enqueue(client, call_id, batch_args, star, kwargs)
                if n_dequeued < n_enqueued:
                    async for output in self._dequeue(client, call_id, n_enqueued - n_dequeued):
                        n_dequeued += 1
                        yield output

    async def __call__(self, *args, **kwargs):
        ''' Uses map, but makes sure there's only 1 output. '''
        outputs = [output async for output in self.map([args], kwargs=kwargs, star=True)]
        assert len(outputs) == 1
        return outputs[0]


def decorate_function(raw_f, image):
    if callable(raw_f):
        return Function(raw_f=raw_f, image=image)
    else:
        raise Exception('%s is not a proper function (of type %s)' % (raw_f, type(raw_f)))
