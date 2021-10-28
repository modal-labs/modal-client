import asyncio
import importlib
import inspect
import os
import sys
import uuid
import warnings

import cloudpickle
from aiostream import stream
from google.protobuf.any_pb2 import Any

from .async_utils import retry, synchronizer
from .buffer_utils import buffered_rpc_read, buffered_rpc_write
from .client import Client
from .config import config, logger
from .exception import RemoteException
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
            # Note: __import__ always returns the top-level package.
            package_path = __import__(module.__package__).__path__
            # TODO: we should handle the array case, https://stackoverflow.com/questions/2699287/what-is-path-useful-for
            assert len(package_path) == 1
            (self.package_path,) = package_path
            self.module_name = module.__spec__.name
            self.recursive_upload = True
            self.remote_dir = "/root/" + module.__package__.split(".")[0]  # TODO: don't hardcode /root
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
        logger.info(f"cwd: {os.getcwd()}")
        logger.info(f"path: {sys.path}")
        logger.info(f"ls: {os.listdir()}")
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


def process_result(client, result):
    if result.status != api_pb2.GenericResult.Status.SUCCESS:
        if result.data:
            try:
                exc = client.deserialize(result.data)
            except Exception:
                exc = None
                warnings.warn("Could not deserialize remote exception!")
            if exc is not None:
                print(result.traceback)
                raise exc
        raise RemoteException(result.exception)

    return client.deserialize(result.data)


@synchronizer
class Invocation:
    def __init__(self, client, function_id, output_buffer_id):
        self.client = client
        self.function_id = function_id
        self.output_buffer_id = output_buffer_id

    @staticmethod
    async def create(function_id, args, kwargs, client):
        request = api_pb2.FunctionMapRequest(function_id=function_id)
        response = await retry(client.stub.FunctionMap)(request)

        input_buffer_id = response.input_buffer_id
        output_buffer_id = response.output_buffer_id

        item = pack_input_buffer_item(client.serialize(args), client.serialize(kwargs), output_buffer_id)
        buffer_req = api_pb2.BufferWriteRequest(item=item, buffer_id=input_buffer_id)
        request = api_pb2.FunctionCallRequest(function_id=function_id, buffer_req=buffer_req)
        await buffered_rpc_write(client.stub.FunctionCall, request)

        return Invocation(client, function_id, output_buffer_id)

    async def run(self):
        async def get_item():
            request = api_pb2.FunctionGetNextOutputRequest(function_id=self.function_id)
            response = await buffered_rpc_read(
                self.client.stub.FunctionGetNextOutput, request, self.output_buffer_id, timeout=None
            )
            return unpack_output_buffer_item(response.item)

        first_result = await get_item()
        if first_result.gen_status != api_pb2.GenericResult.GeneratorStatus.NOT_GENERATOR:

            @synchronizer
            async def gen():
                cur_result = first_result
                while cur_result.gen_status != api_pb2.GenericResult.GeneratorStatus.COMPLETE:
                    yield process_result(self.client, cur_result)

                    cur_result = await get_item()

            return gen()
        else:
            return process_result(self.client, first_result)


@synchronizer
class MapInvocation:
    # TODO: should this be an object?
    def __init__(self, client, response_gen):
        self.client = client
        self.response_gen = response_gen

    @staticmethod
    async def create(function_id, inputs, kwargs, client):
        request = api_pb2.FunctionMapRequest(function_id=function_id)
        response = await retry(client.stub.FunctionMap)(request)

        input_buffer_id = response.input_buffer_id
        output_buffer_id = response.output_buffer_id

        async def pump_inputs():
            async for arg in inputs:
                item = pack_input_buffer_item(client.serialize(arg), client.serialize(kwargs), output_buffer_id)
                buffer_req = api_pb2.BufferWriteRequest(item=item, buffer_id=input_buffer_id)
                request = api_pb2.FunctionCallRequest(function_id=function_id, buffer_req=buffer_req)
                # No timeout so this can block forever.
                yield await buffered_rpc_write(client.stub.FunctionCall, request)
            yield

        async def poll_outputs():
            """Keep trying to dequeue outputs."""
            while True:
                request = api_pb2.FunctionGetNextOutputRequest(function_id=function_id)
                response = await buffered_rpc_read(
                    client.stub.FunctionGetNextOutput, request, output_buffer_id, timeout=None
                )
                yield response

        response_gen = stream.merge(pump_inputs(), poll_outputs())

        return MapInvocation(client, response_gen)

    async def __aiter__(self):
        num_inputs = 0
        have_all_inputs = False
        num_outputs = 0
        async with self.response_gen.stream() as streamer:
            async for response in streamer:
                if response is None:
                    have_all_inputs = True
                elif isinstance(response, api_pb2.BufferWriteResponse):
                    assert not have_all_inputs
                    num_inputs += 1
                elif isinstance(response, api_pb2.BufferReadResponse):
                    result = unpack_output_buffer_item(response.item)

                    if result.gen_status != api_pb2.GenericResult.GeneratorStatus.INCOMPLETE:
                        num_outputs += 1

                    if result.gen_status != api_pb2.GenericResult.GeneratorStatus.COMPLETE:
                        yield process_result(self.client, result)
                else:
                    assert False, f"Got unknown type in invocation stream: {type(response)}"

                assert num_outputs <= num_inputs
                if have_all_inputs and num_outputs == num_inputs:
                    break


@synchronizer
class Function(Object):
    def __init__(self, raw_f, image=None, env_dict=None, client=None):
        assert callable(raw_f)
        self.info = FunctionInfo(raw_f)
        super().__init__(
            args=dict(
                raw_f=raw_f,
                image=image,
                env_dict=env_dict,
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
        if self.args.env_dict is not None:
            env_dict = await self.session.create_or_get_object(self.args.env_dict)
            env_dict_id = env_dict.object_id
        else:
            env_dict_id = None
        mounts = await asyncio.gather(*(self.session.create_or_get_object(mount) for mount in mounts))

        # Create function remotely
        function_definition = api_pb2.Function(
            module_name=self.info.module_name,
            function_name=self.info.function_name,
            mount_ids=[mount.object_id for mount in mounts],
            env_dict_id=env_dict_id,
            image_id=image.object_id,
        )
        request = api_pb2.FunctionGetOrCreateRequest(
            session_id=self.session.session_id,
            function=function_definition,
        )
        response = await self.client.stub.FunctionGetOrCreate(request)
        return response.function_id

    @requires_create
    async def map(self, inputs, window=100, kwargs={}):
        inputs = stream.iterate(inputs)
        inputs = stream.map(inputs, lambda arg: (arg,))
        async for item in await MapInvocation.create(self.object_id, inputs, kwargs, self.client):
            yield item

    @requires_create
    async def __call__(self, *args, **kwargs):
        invocation = await Invocation.create(self.object_id, args, kwargs, self.client)
        return await invocation.run()

    def get_raw_f(self):
        return self.args.raw_f

    @staticmethod
    def get_function(module_name, function_name):
        f = _path_to_function(module_name, function_name)
        assert isinstance(f, Function)
        return f
