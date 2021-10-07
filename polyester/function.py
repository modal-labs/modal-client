import asyncio
import importlib
import inspect
import os
import sys
import uuid

import cloudpickle
from google.protobuf.any_pb2 import Any

from .async_utils import retry, synchronizer
from .buffer_utils import buffered_read_all, buffered_write_all
from .client import Client
from .config import config, logger
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .mount import Mount, create_package_mounts
from .object import Object, requires_create, requires_create
from .proto import api_pb2
from .queue import Queue


def _function_to_path(f):
    function_name = f.__name__
    module = inspect.getmodule(f)
    if module.__package__:
        # This is a "real" module, eg. examples.logs.f
        # Get the package path
        package_path = __import__(module.__package__).__path__
        # TODO: we should handle the array case, https://stackoverflow.com/questions/2699287/what-is-path-useful-for
        assert len(package_path) == 1
        (package_path,) = package_path
        module_name = module.__spec__.name
        recursive_upload = True
        remote_dir = "/root/" + module.__package__  # TODO: don't hardcode /root
    else:
        # This generally covers the case where it's invoked with
        # python foo/bar/baz.py
        module_name = os.path.splitext(os.path.basename(module.__file__))[0]
        package_path = os.path.dirname(module.__file__)
        recursive_upload = False  # Just pick out files in the same directory
        remote_dir = "/root"  # TODO: don't hardcore /root

    # Create mount
    mount = Mount(
        local_dir=package_path,
        remote_dir=remote_dir,
        condition=lambda filename: os.path.splitext(filename)[1] == ".py",
        recursive=recursive_upload,
    )

    return (mount, module_name, function_name)


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
class MapInvocation:
    # TODO: should this be an object?
    def __init__(self, function_id, inputs, kwargs, client, input_buffer_id, output_buffer_id):
        self.function_id = function_id
        self.inputs = inputs
        self.kwargs = kwargs
        self.client = client
        self.input_buffer_id = input_buffer_id
        self.output_buffer_id = output_buffer_id

    @staticmethod
    async def create(function_id, inputs, kwargs, client):
        request = api_pb2.FunctionMapRequest(function_id=function_id)
        response = await retry(client.stub.FunctionMap)(request)
        input_buffer_id = response.input_buffer_id
        output_buffer_id = response.output_buffer_id
        return MapInvocation(function_id, inputs, kwargs, client, input_buffer_id, output_buffer_id)

    async def __aiter__(self):
        async def generate_inputs():
            for arg in iter(self.inputs):
                item = pack_input_buffer_item(
                    self.client.serialize(arg), self.client.serialize(self.kwargs), self.output_buffer_id
                )

                buffer_req = api_pb2.BufferWriteRequest(item=item, buffer_id=self.input_buffer_id)

                yield api_pb2.FunctionCallRequest(function_id=self.function_id, buffer_req=buffer_req)

        # send_EOF is True for now, for easier testing and iteration. Sending this signal also terminates
        # the function container, so we might want to not do that in the future and rely on the timeout instead.
        pump_task = asyncio.create_task(
            buffered_write_all(self.client.stub.FunctionCall, generate_inputs(), send_EOF=True)
        )

        request = api_pb2.FunctionGetNextOutputRequest(function_id=self.function_id)

        async for output in buffered_read_all(self.client.stub.FunctionGetNextOutput, request, self.output_buffer_id):
            result = unpack_output_buffer_item(output)

            if result.status != api_pb2.GenericResult.Status.SUCCESS:
                raise Exception("Remote exception: %s\n%s" % (result.exception, result.traceback))
            yield self.client.deserialize(result.data)

        await asyncio.wait_for(pump_task, timeout=BLOCKING_REQUEST_TIMEOUT)


class Function(Object):
    def __init__(self, raw_f, image=None, client=None):
        assert callable(raw_f)
        super().__init__(
            args=dict(
                raw_f=raw_f,
                image=image,
            ),
        )

    async def _create_or_get(self):
        mount, module_name, function_name = _function_to_path(self.args.raw_f)

        mounts = [mount]
        if config["sync_entrypoint"] and not os.getenv("POLYESTER_IMAGE_LOCAL_ID"):
            # TODO(erikbern): If the first condition is true then we're running in a local
            # client which implies the second is always true as well?
            mounts.extend(create_package_mounts("polyester"))
        # TODO(erikbern): couldn't we just create one single mount with all packages instead of multiple?

        # Wait for image and mounts to finish
        # TODO: should we really join recursively here? Maybe it's better to move this logic to the session class?
        image = await self.session.create_or_get(self.args.image)
        mounts = await asyncio.gather(*(self.session.create_or_get(mount) for mount in mounts))

        # Create function remotely
        function_definition = api_pb2.Function(
            module_name=module_name,
            function_name=function_name,
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
        args = [(arg,) for arg in inputs]
        async for item in await MapInvocation.create(self.object_id, args, kwargs, self.client):
            yield item

    @requires_create
    async def __call__(self, *args, **kwargs):
        invocation = await MapInvocation.create(self.object_id, [args], kwargs, self.client)
        async for output in invocation:
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
        raise Exception("%s is not a proper function (of type %s)" % (raw_f, type(raw_f)))
