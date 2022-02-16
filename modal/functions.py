import asyncio
from typing import Optional

from aiostream import pipe, stream
from google.protobuf.any_pb2 import Any

from ._app_singleton import get_container_app, get_default_app
from ._async_utils import retry
from ._buffer_utils import buffered_rpc_read, buffered_rpc_write
from ._decorator_utils import decorator_with_options
from ._factory import Factory
from ._function_utils import FunctionInfo
from .config import config
from .exception import ExecutionError, InvalidError, RemoteError
from .image import debian_slim
from .mount import Mount, create_package_mounts
from .object import Object
from .proto import api_pb2
from .schedule import Schedule

MODAL_CLIENT_MOUNT_NAME = "modal-client-mount"


# TODO: maybe we can create a special Buffer class in the ORM that keeps track of the protobuf type
# of the bytes stored, so the packing/unpacking can happen automatically.
def _pack_input_buffer_item(
    args: Optional[bytes], kwargs: Optional[bytes], output_buffer_id: str, idx=None
) -> api_pb2.BufferItem:
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


def _process_result(app, result):
    if result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
        if result.data:
            try:
                exc = app.deserialize(result.data)
            except Exception as deser_exc:
                raise ExecutionError(
                    "Could not deserialize remote exception due to local error:\n"
                    + f"{deser_exc}\n"
                    + "This can happen if your local environment does not have the remote exception definitions.\n"
                    + "Here is the remote traceback:\n"
                    + f"{result.traceback}"
                )
            if not isinstance(exc, BaseException):
                raise ExecutionError(f"Got remote exception of incorrect type {type(exc)}")
            print(result.traceback)
            raise exc
        raise RemoteError(result.exception)

    return app.deserialize(result.data)


class _Invocation:
    def __init__(self, app, function_id, output_buffer_id):
        self.app = app
        self.function_id = function_id
        self.output_buffer_id = output_buffer_id

    @staticmethod
    async def create(function_id, args, kwargs, app):
        assert function_id
        request = api_pb2.FunctionMapRequest(function_id=function_id)
        response = await retry(app.client.stub.FunctionMap)(request)

        input_buffer_id = response.input_buffer_id
        output_buffer_id = response.output_buffer_id

        item = _pack_input_buffer_item(app.serialize(args), app.serialize(kwargs), output_buffer_id)
        buffer_req = api_pb2.BufferWriteRequest(items=[item], buffer_id=input_buffer_id)
        request = api_pb2.FunctionCallRequest(function_id=function_id, buffer_req=buffer_req)
        await buffered_rpc_write(app.client.stub.FunctionCall, request)

        return _Invocation(app, function_id, output_buffer_id)

    async def get_items(self):
        request = api_pb2.FunctionGetNextOutputRequest(function_id=self.function_id)
        response = await buffered_rpc_read(
            self.app.client.stub.FunctionGetNextOutput, request, self.output_buffer_id, timeout=None
        )
        for item in response.items:
            yield _unpack_output_buffer_item(item)

    async def run_function(self):
        result = (await stream.list(self.get_items()))[0]
        assert not result.gen_status
        return _process_result(self.app, result)

    async def run_generator(self):
        completed = False
        while not completed:
            async for result in self.get_items():
                if result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                    completed = True
                    break
                yield _process_result(self.app, result)


MAP_INVOCATION_CHUNK_SIZE = 100


class _MapInvocation:
    # TODO: should this be an object?
    def __init__(self, function_id, input_stream, kwargs, app, is_generator):
        self.function_id = function_id
        self.input_stream = input_stream
        self.kwargs = kwargs
        self.app = app
        self.is_generator = is_generator

    async def __aiter__(self):
        request = api_pb2.FunctionMapRequest(function_id=self.function_id)
        response = await retry(self.app.client.stub.FunctionMap)(request)

        input_buffer_id = response.input_buffer_id
        output_buffer_id = response.output_buffer_id

        have_all_inputs = False
        num_outputs = 0
        num_inputs = 0

        async def pump_inputs():
            nonlocal num_inputs, have_all_inputs

            chunked_input_stream = self.input_stream | pipe.chunks(MAP_INVOCATION_CHUNK_SIZE)
            async with chunked_input_stream.stream() as streamer:
                async for chunk in streamer:
                    items = []
                    for arg in chunk:
                        item = _pack_input_buffer_item(
                            self.app.serialize(arg),
                            self.app.serialize(self.kwargs),
                            output_buffer_id,
                            idx=num_inputs,
                        )
                        num_inputs += 1
                        items.append(item)
                    buffer_req = api_pb2.BufferWriteRequest(items=items, buffer_id=input_buffer_id)
                    request = api_pb2.FunctionCallRequest(function_id=self.function_id, buffer_req=buffer_req)

                    await buffered_rpc_write(self.app.client.stub.FunctionCall, request)

            have_all_inputs = True
            yield

        async def poll_outputs():
            nonlocal num_inputs, num_outputs, have_all_inputs

            # map to store out-of-order outputs received
            pending_outputs = {}

            while True:
                request = api_pb2.FunctionGetNextOutputRequest(function_id=self.function_id)
                response = await buffered_rpc_read(
                    self.app.client.stub.FunctionGetNextOutput,
                    request,
                    output_buffer_id,
                    timeout=None,
                    warn_on_cancel=False,
                )
                for item in response.items:
                    result = _unpack_output_buffer_item(item)

                    if self.is_generator:
                        if result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                            num_outputs += 1
                        else:
                            output = _process_result(self.app, result)
                            # yield output directly for generators.
                            yield output
                    else:
                        # hold on to outputs for function maps, so we can reorder them correctly.
                        pending_outputs[item.idx] = _process_result(self.app, result)

                # send outputs sequentially while we can
                while num_outputs in pending_outputs:
                    output = pending_outputs.pop(num_outputs)
                    yield output
                    num_outputs += 1

                if have_all_inputs:
                    assert num_outputs <= num_inputs
                    if num_outputs == num_inputs:
                        break

            assert len(pending_outputs) == 0

        response_gen = stream.merge(pump_inputs(), poll_outputs())

        async with response_gen.stream() as streamer:
            async for response in streamer:
                # Handle yield at the end of pump_inputs, in case
                # that finishes after all outputs have been polled.
                if response is None:
                    if num_outputs == num_inputs:
                        break
                    continue
                yield response


class Function(Object, Factory, type_prefix="fu"):
    def __init__(
        self,
        raw_f,
        image=None,
        secret=None,
        schedule: Optional[Schedule] = None,
        is_generator=False,
        gpu: bool = False,
    ):
        assert callable(raw_f)
        self.info = FunctionInfo(raw_f)
        if schedule is not None:
            if not self.info.is_nullary():
                raise InvalidError(
                    f"Function {raw_f} has a schedule, so it needs to support calling it with no arguments"
                )
        # This is the only place besides object factory that sets tags
        tag = self.info.get_tag(None)
        self.raw_f = raw_f
        self.image = image
        self.secret = secret
        self.schedule = schedule
        self.is_generator = is_generator
        self.gpu = gpu
        super()._init_static(tag=tag)

    async def load(self, app):
        mounts = [
            await Mount.create(
                local_dir=self.info.package_path,
                remote_dir=self.info.remote_dir,
                recursive=self.info.recursive,
                condition=self.info.condition,
            )
        ]
        # TODO(erikbern): couldn't we just create one single mount with all packages instead of multiple?
        if config["sync_entrypoint"]:
            mounts.extend(await create_package_mounts("modal"))
        else:
            client_mount = Mount.include(MODAL_CLIENT_MOUNT_NAME, namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)
            mounts.append(client_mount)

        # Wait for image and mounts to finish
        # TODO: should we really join recursively here? Maybe it's better to move this logic to the app class?
        if self.image is not None:
            image_id = await app.create_object(self.image)
        else:
            image_id = None  # Happens if it's a notebook function
        if self.secret is not None:
            secret_id = await app.create_object(self.secret)
        else:
            secret_id = None
        mount_ids = await asyncio.gather(*(app.create_object(mount) for mount in mounts))

        if self.is_generator:
            function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
        else:
            function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

        # Create function remotely
        function_definition = api_pb2.Function(
            module_name=self.info.module_name,
            function_name=self.info.function_name,
            mount_ids=mount_ids,
            secret_id=secret_id,
            image_id=image_id,
            definition_type=self.info.definition_type,
            function_serialized=self.info.function_serialized,
            function_type=function_type,
            resources=api_pb2.Resources(gpu=self.gpu),
        )
        request = api_pb2.FunctionCreateRequest(
            app_id=app.app_id,
            function=function_definition,
            cron_string=self.schedule._cron_string if self.schedule else None,
            period=self.schedule._period if self.schedule else None,
        )
        response = await app.client.stub.FunctionCreate(request)

        return response.function_id

    async def map(self, inputs, window=100, kwargs={}):
        input_stream = stream.iterate(inputs) | pipe.map(lambda arg: (arg,))
        async for item in _MapInvocation(self.object_id, input_stream, kwargs, self._app, self.is_generator):
            yield item

    async def call_function(self, args, kwargs):
        invocation = await _Invocation.create(self.object_id, args, kwargs, self._app)
        return await invocation.run_function()

    async def invoke_function(self, args, kwargs):
        """Returns a future rather than the result directly"""
        invocation = await _Invocation.create(self.object_id, args, kwargs, self._app)
        return invocation.run_function()

    async def call_generator(self, args, kwargs):
        invocation = await _Invocation.create(self.object_id, args, kwargs, self._app)
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


def _register_function(function, app):
    if get_container_app() is None:
        if app is None:
            app = get_default_app()
        if app is not None:
            app.register_object(function)


@decorator_with_options
def function(
    raw_f=None,
    app=None,
    image=debian_slim,
    schedule: Optional[Schedule] = None,
    secret=None,
    gpu: bool = False,
):
    """Decorator to create Modal functions

    Args:
        app (:py:class:`modal.app.App`): The app
        image (:py:class:`modal.image.Image`): The image to run the function in
        secret (:py:class:`modal.secret.Secret`): Dictionary of environment variables
        gpu (bool): Whether a GPU is required
    """
    function = Function(raw_f, image=image, secret=secret, schedule=schedule, is_generator=False, gpu=gpu)
    _register_function(function, app)
    return function


@decorator_with_options
def generator(raw_f=None, app=None, image=debian_slim, secret=None, gpu=False):
    """Decorator to create Modal generators

    Args:
        app (:py:class:`modal.app.App`): The app
        image (:py:class:`modal.image.Image`): The image to run the function in
        secret (:py:class:`modal.secret.Secret`): Dictionary of environment variables
        gpu (bool): Whether a GPU is required
    """
    function = Function(raw_f, image=image, secret=secret, is_generator=True, gpu=gpu)
    _register_function(function, app)
    return function
