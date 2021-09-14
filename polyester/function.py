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
    function_name = f.__name__
    module = inspect.getmodule(f)
    if module.__package__:
        # This is a "real" module, eg. examples.logs.f
        # Get the package path
        package_path = __import__(module.__package__).__path__
        assert len(package_path) == 1  # TODO: we should handle the array case, https://stackoverflow.com/questions/2699287/what-is-path-useful-for
        package_path, = package_path
        module_name = module.__spec__.name
        recursive_upload = True
    else:
        # This generally covers the case where it's invoked with
        # python foo/bar/baz.py
        module_name = os.path.splitext(os.path.basename(module.__file__))[0]
        package_path = os.path.dirname(module.__file__)
        recursive_upload = False  # Just pick out files in the same directory

    # Create mount
    # TODO: solve this circular import
    from .image import Mount
    mount = Mount(
        local_dir=package_path,
        remote_dir='/root',  # TODO: might be image-dependent, so the image should really supply this to the function
        condition=lambda filename: os.path.splitext(filename)[1] == '.py',
        recursive=recursive_upload,
    )

    return (mount, module_name, function_name)


def _path_to_function(module_name, function_name):
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
        super().__init__(
            client=client,
            args=dict(
                function_id=function_id,
                inputs=inputs,
                star=star,
                window=window,
                kwargs=kwargs,
            ),
        )

    async def _enqueue(self, client, args, star, kwargs):
        if not star:
            # Everything will just be passed as the first input
            args = [(arg,) for arg in args]
        # TODO: break out the creation of the call into a separate request
        request = api_pb2.FunctionCallRequest(
            function_id=self.args.function_id,
            inputs=[client.serialize((arg, kwargs)) for arg in args],
            idempotency_key=str(uuid.uuid4()),
            call_id=self.object_id,
        )
        response = await retry(client.stub.FunctionCall)(request)
        self.object_id = response.call_id

    async def _dequeue(self, client, n_outputs):
        while True:
            request = api_pb2.FunctionGetNextOutputRequest(
                function_id=self.args.function_id,  # TODO: why is this needed?
                call_id=self.object_id,
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
        inputs = iter(self.args.inputs)  # Handle non-generator inputs

        n_enqueued, n_dequeued = 0, 0
        input_exhausted = False
        while not input_exhausted or n_dequeued < n_enqueued:
            logger.debug("Map status: %d enqueued, %d dequeued" % (n_enqueued, n_dequeued))
            batch_args = []
            while not input_exhausted and n_enqueued < n_dequeued + self.args.window:
                try:
                    batch_args.append(next(inputs))
                    n_enqueued += 1
                except StopIteration:
                    input_exhausted = True
            if batch_args:
                await self._enqueue(client, batch_args, self.args.star, self.args.kwargs)
            if n_dequeued < n_enqueued:
                async for output in self._dequeue(client, n_enqueued - n_dequeued):
                    n_dequeued += 1
                    yield output


class Function(Object):
    def __init__(self, raw_f, image=None, client=None):
        assert callable(raw_f)
        super().__init__(
            client=client,
            args=dict(
                raw_f=raw_f,
                image=image,
            ),
        )

    async def _join(self):
        client = await self._get_client()

        mount, module_name, function_name = _function_to_path(self.args.raw_f)

        # await mount.join()

        # Create function remotely
        image_id = await self.args.image.set_client(client).join()
        function_definition = api_pb2.Function(
            module_name=module_name,
            function_name=function_name,
        )
        request = api_pb2.FunctionGetOrCreateRequest(
            session_id=client.session_id,
            image_id=image_id,
            function=function_definition,
        )
        response = await client.stub.FunctionGetOrCreate(request)
        return response.function_id

    async def map(self, inputs, star=False, window=100, kwargs={}):
        client = await self._get_client()
        function_id = await self.join()
        return Call(client, function_id, inputs, star, window, kwargs)

    async def __call__(self, *args, **kwargs):
        ''' Uses map, but makes sure there's only 1 output. '''
        async for output in self.map([args], kwargs=kwargs, star=True):
            return output  # return the first (and only) one

    @staticmethod
    def get_function(module_name, function_name):
        f = _path_to_function(module_name, function_name)
        assert isinstance(f, Function)
        return f.args.raw_f


def decorate_function(raw_f, image):
    if callable(raw_f):
        return Function(raw_f=raw_f, image=image)
    else:
        raise Exception('%s is not a proper function (of type %s)' % (raw_f, type(raw_f)))
