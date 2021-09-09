import asyncio
import cloudpickle
import importlib
import inspect
import os
import sys
import uuid

from .async_utils import retry, synchronizer
from .client import Client
from .config import logger
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .proto import api_pb2
from .object import Object
from .queue import Queue


def _function_to_path(f):
    module = inspect.getmodule(f)
    if module.__spec__:
        # TODO: even if the module has a spec, we should probably check the file of it
        # Either we can infer that it's a system module, or if it's something in the
        # cwd then let's make sure it can be imported?
        module_name = module.__spec__.name
    else:
        # Note that this case covers both these cases:
        # python -m foo.bar
        # python foo/bar.py
        fn = os.path.splitext(module.__file__)[0]
        path = os.path.relpath(fn, os.getcwd())  # TODO: might want to do it relative to other paths in sys.path?
        parts = os.path.split(path)
        module_name = '.'.join(p for p in parts if p and not p.startswith('.'))
    function_name = f.__name__
    return (module_name, function_name)


def _path_to_function(module_name, function_name):
    # Opposite of _function_to_path
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except ModuleNotFoundError:
        # Just print some debug stuff, then re-raise
        logger.info(f'{os.getcwd()=}')
        logger.info(f'{sys.path=}')
        logger.info(f'{os.listdir()=}')
        raise


class Call(Object):
    def __init__(self, client, function_id, inputs, star, window, kwargs):
        super().__init__(client=client)  # TODO: tag, id
        self.function_id = function_id
        self.inputs = inputs
        self.star = star
        self.window = window
        self.kwargs = kwargs
        self.call_id = None

    async def _enqueue(self, client, args, star, kwargs):
        if not star:
            # Everything will just be passed as the first input
            args = [(arg,) for arg in args]
        request = api_pb2.FunctionCallRequest(
            function_id=self.function_id,
            inputs=[client.serialize((arg, kwargs)) for arg in args],
            idempotency_key=str(uuid.uuid4()),
            call_id=self.call_id
        )
        response = await retry(client.stub.FunctionCall)(request)
        self.call_id = response.call_id

    async def _dequeue(self, client, n_outputs):
        while True:
            request = api_pb2.FunctionGetNextOutputRequest(
                function_id=self.function_id,
                call_id=self.call_id,
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

    async def __aiter__(self):
        # Most of the complexity of this function comes from the input throttling.
        # Basically the idea is that we maintain x (default 100) outstanding requests at any point in time,
        # and we don't enqueue more requests until we get back enough values.
        # It probably makes a lot of sense to move the input throttling to the server instead.

        client = await self._get_client()

        # TODO: we should support asynchronous generators as well
        inputs = iter(self.inputs)  # Handle non-generator inputs

        n_enqueued, n_dequeued = 0, 0
        input_exhausted = False
        while not input_exhausted or n_dequeued < n_enqueued:
            logger.debug('Map status: %d enqueued, %d dequeued' % (n_enqueued, n_dequeued))
            batch_args = []
            while not input_exhausted and n_enqueued < n_dequeued + self.window:
                try:
                    batch_args.append(next(inputs))
                    n_enqueued += 1
                except StopIteration:
                    input_exhausted = True
            if batch_args:
                await self._enqueue(client, batch_args, self.star, self.kwargs)
            if n_dequeued < n_enqueued:
                async for output in self._dequeue(client, n_enqueued - n_dequeued):
                    n_dequeued += 1
                    yield output


class Function(Object):
    def __init__(self, raw_f, image=None, client=None):
        super().__init__(client=client)  # TODO: tag, id
        assert callable(raw_f)
        self.raw_f = raw_f
        self.module_name, self.function_name = _function_to_path(raw_f)
        self.image = image
        self.function_id = None

    async def _get_id(self):
        if self.function_id:
            return self.function_id

        client = await self._get_client()

        # Create function remotely
        image_id = await self.image.join(client)
        function_definition = api_pb2.Function(
            module_name=self.module_name,
            function_name=self.function_name,
        )
        request = api_pb2.FunctionGetOrCreateRequest(
            session_id=client.session_id,
            image_id=image_id,
            function=function_definition,
        )
        response = await client.stub.FunctionGetOrCreate(request)
        self.function_id = response.function_id
        return self.function_id

    async def map(self, inputs, star=False, window=100, kwargs={}):
        client = await self._get_client()
        function_id = await self._get_id()
        return Call(client, function_id, inputs, star, window, kwargs)

    async def __call__(self, *args, **kwargs):
        ''' Uses map, but makes sure there's only 1 output. '''
        async for output in self.map([args], kwargs=kwargs, star=True):
            return output  # return the first (and only) one

    @staticmethod
    def get_function(module_name, function_name):
        f = _path_to_function(module_name, function_name)
        assert isinstance(f, Function)
        return f.raw_f


def decorate_function(raw_f, image):
    if callable(raw_f):
        return Function(raw_f=raw_f, image=image)
    else:
        raise Exception('%s is not a proper function (of type %s)' % (raw_f, type(raw_f)))
