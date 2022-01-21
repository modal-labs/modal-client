import asyncio
import os
import sys
import uuid
import warnings

from aiostream import pipe, stream
from google.protobuf.any_pb2 import Any

from ._async_utils import retry
from ._buffer_utils import buffered_rpc_read, buffered_rpc_write
from ._decorator_utils import decorator_with_options
from ._factory import Factory
from ._function_utils import FunctionInfo
from ._session_singleton import get_container_session, get_default_session
from .config import config, logger
from .exception import RemoteError
from .image import debian_slim
from .mount import Mount, create_package_mounts
from .object import Object
from .proto import api_pb2


# TODO: maybe we can create a special Buffer class in the ORM that keeps track of the protobuf type
# of the bytes stored, so the packing/unpacking can happen automatically.
def _pack_input_buffer_item(args: bytes, kwargs: bytes, output_buffer_id: str, idx=None) -> api_pb2.BufferItem:
    data = Any()
    data.Pack(api_pb2.FunctionInput(args=args, kwargs=kwargs, output_buffer_id=output_buffer_id))
    return api_pb2.BufferItem(data=data, idx=idx)


def _pack_output_buffer_item(result: api_pb2.GenericResult, idx=None) -> api_pb2.BufferItem:
    data = Any()
    data.Pack(result)
    return api_pb2.BufferItem(data=data, idx=idx)


def _unpack_input_buffer_item(buffer_item: api_pb2.BufferItem) -> api_pb2.FunctionInput:
    input = api_pb2.FunctionInput()
    buffer_item.data.Unpack(input)
    return input


def _unpack_output_buffer_item(buffer_item: api_pb2.BufferItem) -> api_pb2.GenericResult:
    output = api_pb2.GenericResult()
    buffer_item.data.Unpack(output)
    return output


def _process_result(session, result):
    if result.status != api_pb2.GenericResult.Status.SUCCESS:
        if result.data:
            try:
                exc = session.deserialize(result.data)
            except Exception:
                exc = None
                warnings.warn("Could not deserialize remote exception!")
            if exc is not None:
                print(result.traceback)
                raise exc
        raise RemoteError(result.exception)

    return session.deserialize(result.data)


class _Invocation:
    def __init__(self, session, function_id, output_buffer_id):
        self.session = session
        self.function_id = function_id
        self.output_buffer_id = output_buffer_id

    @staticmethod
    async def create(function_id, args, kwargs, session):
        assert function_id
        request = api_pb2.FunctionMapRequest(function_id=function_id)
        response = await retry(session.client.stub.FunctionMap)(request)

        input_buffer_id = response.input_buffer_id
        output_buffer_id = response.output_buffer_id

        item = _pack_input_buffer_item(session.serialize(args), session.serialize(kwargs), output_buffer_id)
        buffer_req = api_pb2.BufferWriteRequest(items=[item], buffer_id=input_buffer_id)
        request = api_pb2.FunctionCallRequest(function_id=function_id, buffer_req=buffer_req)
        await buffered_rpc_write(session.client.stub.FunctionCall, request)

        return _Invocation(session, function_id, output_buffer_id)

    async def get_items(self):
        request = api_pb2.FunctionGetNextOutputRequest(function_id=self.function_id)
        response = await buffered_rpc_read(
            self.session.client.stub.FunctionGetNextOutput, request, self.output_buffer_id, timeout=None
        )
        for item in response.items:
            yield _unpack_output_buffer_item(item)

    async def run_function(self):
        result = (await stream.list(self.get_items()))[0]
        assert result.gen_status == api_pb2.GenericResult.GeneratorStatus.NOT_GENERATOR
        return _process_result(self.session, result)

    async def run_generator(self):
        completed = False
        while not completed:
            async for result in self.get_items():
                if result.gen_status == api_pb2.GenericResult.GeneratorStatus.COMPLETE:
                    completed = True
                    break
                yield _process_result(self.session, result)


MAP_INVOCATION_CHUNK_SIZE = 100


class _MapInvocation:
    # TODO: should this be an object?
    def __init__(self, session, response_gen):
        self.session = session
        self.response_gen = response_gen

    @staticmethod
    async def create(function_id, input_stream, kwargs, session, is_generator):
        request = api_pb2.FunctionMapRequest(function_id=function_id)
        response = await retry(session.client.stub.FunctionMap)(request)

        input_buffer_id = response.input_buffer_id
        output_buffer_id = response.output_buffer_id

        async def pump_inputs():
            chunked_input_stream = input_stream | pipe.chunks(MAP_INVOCATION_CHUNK_SIZE)
            last_idx_sent = -1
            async with chunked_input_stream.stream() as streamer:
                async for chunk in streamer:
                    items = []
                    for arg in chunk:
                        last_idx_sent += 1
                        item = _pack_input_buffer_item(
                            session.serialize(arg), session.serialize(kwargs), output_buffer_id, idx=last_idx_sent
                        )
                        items.append(item)
                    buffer_req = api_pb2.BufferWriteRequest(items=items, buffer_id=input_buffer_id)
                    request = api_pb2.FunctionCallRequest(function_id=function_id, buffer_req=buffer_req)

                    response = await buffered_rpc_write(session.client.stub.FunctionCall, request)
                    yield len(items)
            yield

        async def poll_outputs():
            """Keep trying to dequeue outputs."""

            next_idx = 0
            # map to store out-of-order outputs received
            pending_outputs = {}

            while True:
                request = api_pb2.FunctionGetNextOutputRequest(function_id=function_id)
                response = await buffered_rpc_read(
                    session.client.stub.FunctionGetNextOutput, request, output_buffer_id, timeout=None
                )
                for item in response.items:
                    assert item.idx >= next_idx

                    # yield output directly for generators.
                    if is_generator:
                        yield item
                    # hold on to outputs for function maps, so we can reorder them correctly.
                    else:
                        pending_outputs[item.idx] = item

                # send outputs sequentially while we can
                while next_idx in pending_outputs:
                    item = pending_outputs.pop(next_idx)
                    yield item
                    next_idx += 1

            assert len(pending_outputs) == 0

        response_gen = stream.merge(pump_inputs(), poll_outputs())

        return _MapInvocation(session, response_gen)

    async def __aiter__(self):
        num_inputs = 0
        have_all_inputs = False
        num_outputs = 0
        async with self.response_gen.stream() as streamer:
            async for response in streamer:
                if response is None:
                    have_all_inputs = True
                elif isinstance(response, int):
                    assert not have_all_inputs
                    num_inputs += response
                elif isinstance(response, api_pb2.BufferItem):
                    result = _unpack_output_buffer_item(response)

                    if result.gen_status != api_pb2.GenericResult.GeneratorStatus.INCOMPLETE:
                        num_outputs += 1

                    if result.gen_status != api_pb2.GenericResult.GeneratorStatus.COMPLETE:
                        yield _process_result(self.session, result)
                else:
                    assert False, f"Got unknown type in invocation stream: {type(response)}"

                if have_all_inputs:
                    assert num_outputs <= num_inputs
                    if num_outputs == num_inputs:
                        break


class Function(Object, Factory, type_prefix="fu"):
    def __init__(self, raw_f, image=None, env_dict=None, schedule=None, is_generator=False, gpu=False):
        assert callable(raw_f)
        self.info = FunctionInfo(raw_f)
        if schedule is not None:
            assert self.info.is_nullary()
        # This is the only place besides object factory that sets tags
        tag = self.info.get_tag(None)
        self.raw_f = raw_f
        self.image = image
        self.env_dict = env_dict
        self.schedule = schedule
        self.is_generator = is_generator
        self.gpu = gpu
        super()._init_static(tag=tag)

    async def load(self, session):
        mounts = [
            await Mount.create(
                local_dir=self.info.package_path,
                remote_dir=self.info.remote_dir,
                recursive=self.info.recursive,
                condition=self.info.condition,
            )
        ]
        if config["sync_entrypoint"]:
            mounts.extend(await create_package_mounts("modal"))
        # TODO(erikbern): couldn't we just create one single mount with all packages instead of multiple?

        # Wait for image and mounts to finish
        # TODO: should we really join recursively here? Maybe it's better to move this logic to the session class?
        if self.image is not None:
            image_id = await session.create_object(self.image)
        else:
            image_id = None  # Happens if it's a notebook function
        if self.env_dict is not None:
            env_dict_id = await session.create_object(self.env_dict)
        else:
            env_dict_id = None
        mount_ids = await asyncio.gather(*(session.create_object(mount) for mount in mounts))

        if self.schedule is not None:
            # Ensure that the function does not require any input arguments
            schedule_id = await session.create_object(self.schedule)
        else:
            schedule_id = None

        if self.is_generator:
            function_type = api_pb2.Function.FunctionType.GENERATOR
        else:
            function_type = api_pb2.Function.FunctionType.FUNCTION

        # Create function remotely
        function_definition = api_pb2.Function(
            module_name=self.info.module_name,
            function_name=self.info.function_name,
            mount_ids=mount_ids,
            env_dict_id=env_dict_id,
            image_id=image_id,
            definition_type=self.info.definition_type,
            function_serialized=self.info.function_serialized,
            function_type=function_type,
            resources=api_pb2.Resources(gpu=self.gpu),
        )
        request = api_pb2.FunctionGetOrCreateRequest(
            session_id=session.session_id,
            schedule_id=schedule_id,
            function=function_definition,
        )
        response = await session.client.stub.FunctionGetOrCreate(request)

        return response.function_id

    async def map(self, inputs, window=100, kwargs={}):
        input_stream = stream.iterate(inputs) | pipe.map(lambda arg: (arg,))
        async for item in await _MapInvocation.create(
            self.object_id, input_stream, kwargs, self._session, self.is_generator
        ):
            yield item

    async def call_function(self, args, kwargs):
        invocation = await _Invocation.create(self.object_id, args, kwargs, self._session)
        return await invocation.run_function()

    async def invoke_function(self, args, kwargs):
        """Returns a future rather than the result directly"""
        invocation = await _Invocation.create(self.object_id, args, kwargs, self._session)
        return invocation.run_function()

    async def call_generator(self, args, kwargs):
        invocation = await _Invocation.create(self.object_id, args, kwargs, self._session)
        async for res in invocation.run_generator():
            yield res

    def __call__(self, *args, **kwargs):
        if self.is_generator:
            return self.call_generator(args, kwargs)
        else:
            return self.call_function(args, kwargs)

    def invoke(self, *args, **kwargs):
        if self.is_generator:
            return self.call_generator(args, kwargs)
        else:
            return self.invoke_function(args, kwargs)

    def get_raw_f(self):
        return self.raw_f


def _register_function(function, session):
    if get_container_session() is None:
        if session is None:
            session = get_default_session()
        if session is not None:
            session.register_object(function)


@decorator_with_options
def function(raw_f=None, session=None, image=debian_slim, schedule=None, env_dict=None, gpu=False):
    """Decorator to create Modal functions

    Args:
        session (:py:class:`modal.session.Session`): The session
        image (:py:class:`modal.image.Image`): The image to run the function in
        env_dict (:py:class:`modal.env_dict.EnvDict`): Dictionary of environment variables
        gpu (bool): Whether a GPU is required
    """
    function = Function(raw_f, image=image, env_dict=env_dict, schedule=schedule, is_generator=False, gpu=gpu)
    _register_function(function, session)
    return function


@decorator_with_options
def generator(raw_f=None, session=None, image=debian_slim, env_dict=None, gpu=False):
    """Decorator to create Modal generators

    Args:
        session (:py:class:`modal.session.Session`): The session
        image (:py:class:`modal.image.Image`): The image to run the function in
        env_dict (:py:class:`modal.env_dict.EnvDict`): Dictionary of environment variables
        gpu (bool): Whether a GPU is required
    """
    function = Function(raw_f, image=image, env_dict=env_dict, is_generator=True, gpu=gpu)
    _register_function(function, session)
    return function
