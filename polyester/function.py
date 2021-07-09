import asyncio
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


def _get_function_name(obj):
    # This isn't used for anything other than a human-readable label,
    # so we can be pretty opportunistic
    module = inspect.getmodule(obj)
    if module.__spec__:
        module_name = module.__spec__.name
    else:
        fn = os.path.splitext(module.__file__)[0]
        path = os.path.relpath(fn, os.getcwd())
        parts = os.path.split(path)
        module_name = '.'.join(p for p in parts if p and not p.startswith('.'))
    function_name = obj.__name__
    return '%s.%s' % (module_name, function_name)
    

@serializable
@synchronizer
class Function:
    def __init__(self, raw_f=None, image=None, client=None, local_id=None, remote_id=None):
        if raw_f is not None:
            assert callable(raw_f)
            self.raw_f = raw_f
            self.name = _get_function_name(raw_f)
        self.image = image
        super().__init__(client=client, local_id=local_id, remote_id=remote_id)

    async def _get_client(self):
        # TODO: move this to sit on SerializableObject?
        if self.client is None:
            self.client = Client()  # TODO: reuse
            await self.client.start()
        return self.client

    async def map(self, inputs, unordered=False, star=False, return_exceptions=False, window=100, kwargs={}):
        # Most of the complexity of this function comes from the input throttling.
        # Basically the idea is that we maintain x (default 100) outstanding requests at any point in time,
        # and we don't enqueue more requests until we get back enough values.
        client = await self._get_client()

        if not self.remote_id:
            # TODO: implement function disambiguation later
            #request = api_pb2.FunctionExistsRequest(client_id=client.id, client_side_key=self.client_side_key)
            #response = await client.stub.FunctionExists(request)

            image_id = await self.image.start(client)
            await self.image.join(client)
            data = client.serialize(self.raw_f)
            request = api_pb2.FunctionCreateRequest(client_id=client.client_id, data=data,
                                                    image_id=image_id, name=self.name)
            response = await client.stub.FunctionCreate(request)
            self.remote_id = response.function_id

        # TODO: we should support asynchronous generators as well
        inputs = iter(inputs)  # Handle non-generator inputs
        call_id = None

        async def enqueue(args):
            nonlocal call_id
            if not star:
                # Everything will just be passed as the first input
                args = [(arg,) for arg in args]
            request = api_pb2.FunctionCallRequest(
                function_id=self.remote_id,
                inputs=[client.serialize((arg, kwargs)) for arg in args],
                idempotency_key=str(uuid.uuid4()),
                call_id=call_id
            )
            response = await retry(client.stub.FunctionCall)(request)
            call_id = response.call_id

        async def dequeue(n_outputs):
            while True:
                request = api_pb2.FunctionGetNextOutputRequest(
                    function_id=self.remote_id,
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
                await enqueue(batch_args)
            if n_dequeued < n_enqueued:
                async for output in dequeue(n_enqueued - n_dequeued):
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
