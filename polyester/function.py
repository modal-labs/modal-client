import aiostream
import asyncio
import importlib
import inspect
import os
import sys
import uuid

import cloudpickle
from google.protobuf.any_pb2 import Any

from .async_utils import retry, synchronizer, create_task
from .buffer_utils import buffered_read_all, buffered_write_all
from .client import Client
from .config import config, logger
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .mount import Mount, create_package_mounts
from .object import Object, requires_create
from .proto import api_pb2
from .queue import Queue


class FunctionInfo:
    def __init__(self, f):
        self.function_name = f.__name__
        module = inspect.getmodule(f)
        if module.__package__:
            # This is a "real" module, eg. examples.logs.f
            # Get the package path
            package_path = __import__(module.__package__).__path__
            # TODO: we should handle the array case, https://stackoverflow.com/questions/2699287/what-is-path-useful-for
            assert len(package_path) == 1
            (self.package_path,) = package_path
            self.module_name = module.__spec__.name
            self.recursive_upload = True
            self.remote_dir = "/root/" + module.__package__  # TODO: don't hardcode /root
        else:
            # This generally covers the case where it's invoked with
            # python foo/bar/baz.py
            self.module_name = os.path.splitext(os.path.basename(module.__file__))[0]
            self.package_path = os.path.dirname(module.__file__)
            self.recursive_upload = False  # Just pick out files in the same directory
            self.remote_dir = "/root"  # TODO: don't hardcore /root

    def get_mount(self):
        return Mount(
            local_dir=self.package_path,
            remote_dir=self.remote_dir,
            condition=lambda filename: os.path.splitext(filename)[1] == ".py",
            recursive=self.recursive_upload,
        )


def _path_to_function(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except ModuleNotFoundError:
        # Just print some debug stuff, then re-raise
        logger.info(f"{os.getcwd()=}")
        logger.info(f"{sys.path=}")
        logger.info(f"{os.listdir()=}")
        raise


# TODO: maybe we can create a special Buffer class in the ORM that keeps track of the protobuf type
# of the bytes stored, so the packing/unpacking can happen automatically.
def pack_input_buffer_item(args: bytes, kwargs: bytes, output_buffer_id: str) -> api_pb2.BufferItem:
    data = Any()
    data.Pack(api_pb2.FunctionInput(args=args, kwargs=kwargs, output_buffer_id=output_buffer_id))
    return api_pb2.BufferItem(data=data)


def pack_output_buffer_item(result: api_pb2.GenericResult) -> api_pb2.BufferItem:
    data = Any()
    data.Pack(result)
    return api_pb2.BufferItem(data=data)


def unpack_input_buffer_item(buffer_item: api_pb2.BufferItem) -> api_pb2.FunctionInput:
    input = api_pb2.FunctionInput()
    buffer_item.data.Unpack(input)
    return input


def unpack_output_buffer_item(buffer_item: api_pb2.BufferItem) -> api_pb2.GenericResult:
    output = api_pb2.GenericResult()
    buffer_item.data.Unpack(output)
    return output


@synchronizer
class Invocation:
    # TODO: should this be an object?
    def __init__(self, client, pump_task, output_generator):
        self.client = client
        self.pump_task = pump_task
        self.output_generator = output_generator

    @staticmethod
    async def create(function_id, inputs, kwargs, client):
        request = api_pb2.FunctionMapRequest(function_id=function_id)
        response = await retry(client.stub.FunctionMap)(request)
        input_buffer_id = response.input_buffer_id
        output_buffer_id = response.output_buffer_id

        def get_protobuf(arg):
            item = pack_input_buffer_item(client.serialize(arg), client.serialize(kwargs), output_buffer_id)
            buffer_req = api_pb2.BufferWriteRequest(item=item, buffer_id=input_buffer_id)
            return api_pb2.FunctionCallRequest(function_id=function_id, buffer_req=buffer_req)

        inputs = aiostream.stream.map(inputs, get_protobuf)

        # send_EOF is True for now, for easier testing and iteration. Sending this signal also terminates
        # the function container, so we might want to not do that in the future and rely on the timeout instead.
        pump_task = create_task(buffered_write_all(client.stub.FunctionCall, inputs, send_EOF=True))

        request = api_pb2.FunctionGetNextOutputRequest(function_id=function_id)

        output_generator = buffered_read_all(client.stub.FunctionGetNextOutput, request, output_buffer_id)

        return Invocation(client, pump_task, output_generator)

    def process_result(self, result):
        if result.status != api_pb2.GenericResult.Status.SUCCESS:
            exc = self.client.deserialize(result.exception)
            raise exc

        return self.client.deserialize(result.data)

    async def peek(self):
        """Get the next output from the iterator. Not named __anext__ because it returns the raw output,
        and not the deserialized data that the main iterator returns."""

        output = await self.output_generator.__anext__()
        return unpack_output_buffer_item(output)

    async def __aiter__(self):
        async for output in self.output_generator:
            result = unpack_output_buffer_item(output)

            if result.gen_status == api_pb2.GenericResult.GeneratorStatus.COMPLETE:
                continue

            yield self.process_result(result)

        await asyncio.wait_for(self.pump_task, timeout=BLOCKING_REQUEST_TIMEOUT)


@synchronizer
class Function(Object):
    def __init__(self, raw_f, image=None, client=None):
        assert callable(raw_f)
        self.info = FunctionInfo(raw_f)
        super().__init__(
            args=dict(
                raw_f=raw_f,
                image=image,
            ),
        )

    async def create_or_get(self):
        mounts = [self.info.get_mount()]
        if config["sync_entrypoint"] and not os.getenv("POLYESTER_IMAGE_LOCAL_ID"):
            # TODO(erikbern): If the first condition is true then we're running in a local
            # client which implies the second is always true as well?
            mounts.extend(create_package_mounts("polyester"))
        # TODO(erikbern): couldn't we just create one single mount with all packages instead of multiple?

        # Wait for image and mounts to finish
        # TODO: should we really join recursively here? Maybe it's better to move this logic to the session class?
        image = await self.session.create_or_get_object(self.args.image)
        mounts = await asyncio.gather(*(self.session.create_or_get_object(mount) for mount in mounts))

        # Create function remotely
        function_definition = api_pb2.Function(
            module_name=self.info.module_name,
            function_name=self.info.function_name,
            mount_ids=[mount.object_id for mount in mounts],
        )
        request = api_pb2.FunctionGetOrCreateRequest(
            session_id=self.session.session_id,
            image_id=image.object_id,  # TODO: move into the function definition?
            function=function_definition,
        )
        response = await self.client.stub.FunctionGetOrCreate(request)
        return response.function_id

    @requires_create
    async def map(self, inputs, window=100, kwargs={}):
        inputs = aiostream.stream.iterate(inputs)
        inputs = aiostream.stream.map(inputs, lambda arg: (arg,))
        async for item in await Invocation.create(self.object_id, inputs, kwargs, self.client):
            yield item

    @requires_create
    async def __call__(self, *args, **kwargs):
        inputs = aiostream.stream.iterate([args])
        invocation = await Invocation.create(self.object_id, inputs, kwargs, self.client)

        # dumb but we need to pop a value from the iterator to see if it's incomplete.
        first_result = await invocation.peek()

        if first_result.gen_status != api_pb2.GenericResult.GeneratorStatus.NOT_GENERATOR:

            @synchronizer
            async def gen():
                if first_result.gen_status != api_pb2.GenericResult.GeneratorStatus.COMPLETE:
                    yield invocation.process_result(first_result)
                async for result in invocation:
                    yield result

            return gen()
        else:
            return invocation.process_result(first_result)

    def get_raw_f(self):
        return self.args.raw_f

    @staticmethod
    def get_function(module_name, function_name):
        f = _path_to_function(module_name, function_name)
        assert isinstance(f, Function)
        return f
