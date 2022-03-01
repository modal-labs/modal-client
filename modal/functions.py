import asyncio
from typing import Collection, Optional

from aiostream import pipe, stream

from ._app_singleton import get_container_app, get_default_app
from ._async_utils import retry
from ._buffer_utils import buffered_rpc_read, buffered_rpc_write
from ._decorator_utils import decorator_with_options
from ._factory import Factory
from ._function_utils import FunctionInfo
from .config import config
from .exception import ExecutionError, InvalidError, NotFoundError, RemoteError
from .image import debian_slim
from .mount import Mount, create_package_mounts
from .object import Object
from .proto import api_pb2
from .rate_limit import RateLimit
from .schedule import Schedule
from .secret import Secret

MODAL_CLIENT_MOUNT_NAME = "modal-client-mount"


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
    def __init__(self, app, function_id, function_call_id):
        self.app = app
        self.function_id = function_id
        self.function_call_id = function_call_id

    @staticmethod
    async def create(function_id, args, kwargs, app):
        assert function_id
        request = api_pb2.FunctionMapRequest(function_id=function_id)
        response = await retry(app.client.stub.FunctionMap)(request)

        function_call_id = response.function_call_id

        inp = api_pb2.FunctionInput(
            args=app.serialize(args), kwargs=app.serialize(kwargs), function_call_id=function_call_id
        )
        request_put = api_pb2.FunctionPutInputsRequest(function_id=function_id, inputs=[inp])
        await buffered_rpc_write(app.client.stub.FunctionPutInputs, request_put)

        return _Invocation(app, function_id, function_call_id)

    async def get_items(self):
        request = api_pb2.FunctionGetOutputsRequest(function_call_id=self.function_call_id)
        response = await buffered_rpc_read(self.app.client.stub.FunctionGetOutputs, request, timeout=None)
        for output in response.outputs:
            yield output

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

        function_call_id = response.function_call_id

        have_all_inputs = False
        num_outputs = 0
        num_inputs = 0

        async def pump_inputs():
            nonlocal num_inputs, have_all_inputs

            chunked_input_stream = self.input_stream | pipe.chunks(MAP_INVOCATION_CHUNK_SIZE)
            async with chunked_input_stream.stream() as streamer:
                async for chunk in streamer:
                    inputs = []
                    for arg in chunk:
                        function_input = api_pb2.FunctionInput(
                            args=self.app.serialize(arg),
                            kwargs=self.app.serialize(self.kwargs),
                            function_call_id=function_call_id,
                            idx=num_inputs,
                        )
                        num_inputs += 1
                        inputs.append(function_input)

                    request = api_pb2.FunctionPutInputsRequest(function_id=self.function_id, inputs=inputs)

                    await buffered_rpc_write(self.app.client.stub.FunctionPutInputs, request)

            have_all_inputs = True
            yield

        async def poll_outputs():
            nonlocal num_inputs, num_outputs, have_all_inputs

            # map to store out-of-order outputs received
            pending_outputs = {}

            while True:
                request = api_pb2.FunctionGetOutputsRequest(function_call_id=function_call_id)
                response = await buffered_rpc_read(
                    self.app.client.stub.FunctionGetOutputs, request, timeout=None, warn_on_cancel=False
                )

                for result in response.outputs:
                    if self.is_generator:
                        if result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                            num_outputs += 1
                        else:
                            output = _process_result(self.app, result)
                            # yield output directly for generators.
                            yield output
                    else:
                        # hold on to outputs for function maps, so we can reorder them correctly.
                        pending_outputs[result.idx] = _process_result(self.app, result)

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
        secret: Optional[Secret] = None,
        secrets: Collection[Secret] = (),
        schedule: Optional[Schedule] = None,
        is_generator=False,
        gpu: bool = False,
        rate_limit: Optional[RateLimit] = None,
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
        if secret and secrets:
            raise InvalidError(f"Function {raw_f} has both singular `secret` and plural `secrets` attached")
        if secret:
            self.secrets: Collection[Secret] = [secret]
        else:
            self.secrets = secrets
        self.schedule = schedule
        self.is_generator = is_generator
        self.gpu = gpu
        self.rate_limit = rate_limit
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
        secret_ids = []
        for secret in self.secrets:
            try:
                secret_id = await app.create_object(secret)
            except NotFoundError as ex:
                raise NotFoundError(
                    f"Could not find secret {ex.obj_repr}\n"
                    + "You can add secrets to your account at https://modal.com/secrets",
                    ex.obj_repr,
                )
            secret_ids.append(secret_id)

        mount_ids = await asyncio.gather(*(app.create_object(mount) for mount in mounts))

        if self.is_generator:
            function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
        else:
            function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

        rate_limit = self.rate_limit.to_proto() if self.rate_limit else None

        # Create function remotely
        function_definition = api_pb2.Function(
            module_name=self.info.module_name,
            function_name=self.info.function_name,
            mount_ids=mount_ids,
            secret_ids=secret_ids,
            image_id=image_id,
            definition_type=self.info.definition_type,
            function_serialized=self.info.function_serialized,
            function_type=function_type,
            resources=api_pb2.Resources(gpu=self.gpu),
            rate_limit=rate_limit,
        )
        request = api_pb2.FunctionCreateRequest(
            app_id=app.app_id,
            function=function_definition,
            schedule=self.schedule.proto_message,
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

    async def call_function_nowait(self, args, kwargs):
        await _Invocation.create(self.object_id, args, kwargs, self._app)

    async def call_generator(self, args, kwargs):
        invocation = await _Invocation.create(self.object_id, args, kwargs, self._app)
        async for res in invocation.run_generator():
            yield res

    async def call_generator_nowait(self, args, kwargs):
        await _Invocation.create(self.object_id, args, kwargs, self._app)

    def __call__(self, *args, **kwargs):
        if self.is_generator:
            return self.call_generator(args, kwargs)
        else:
            return self.call_function(args, kwargs)

    async def enqueue(self, *args, **kwargs):
        """Calls the function with the given arguments without waiting for the results"""
        if self.is_generator:
            await self.call_generator_nowait(args, kwargs)
        else:
            await self.call_function_nowait(args, kwargs)

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
    secret: Optional[Secret] = None,
    secrets: Collection[Secret] = (),
    gpu: bool = False,
    rate_limit: Optional[RateLimit] = None,
):
    """Decorator to create Modal functions

    Args:
        app (:py:class:`modal.app.App`): The app
        image (:py:class:`modal.image.Image`): The image to run the function in
        secret (:py:class:`modal.secret.Secret`): Dictionary of environment variables
        gpu (bool): Whether a GPU is required
    """
    function = Function(
        raw_f,
        image=image,
        secret=secret,
        secrets=secrets,
        schedule=schedule,
        is_generator=False,
        gpu=gpu,
        rate_limit=rate_limit,
    )
    _register_function(function, app)
    return function


@decorator_with_options
def generator(
    raw_f=None,
    app=None,
    image=debian_slim,
    secret: Optional[Secret] = None,
    secrets: Collection[Secret] = (),
    gpu: bool = False,
    rate_limit: Optional[RateLimit] = None,
):
    """Decorator to create Modal generators

    Args:
        app (:py:class:`modal.app.App`): The app
        image (:py:class:`modal.image.Image`): The image to run the function in
        secret (:py:class:`modal.secret.Secret`): Dictionary of environment variables
        gpu (bool): Whether a GPU is required
    """
    function = Function(
        raw_f, image=image, secret=secret, secrets=secrets, is_generator=True, gpu=gpu, rate_limit=rate_limit
    )
    _register_function(function, app)
    return function
