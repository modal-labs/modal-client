import asyncio
from typing import Collection, Optional

from aiostream import pipe, stream

from modal_proto import api_pb2
from modal_utils.async_utils import queue_batch_iterator, retry, synchronize_apis

from ._blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from ._buffer_utils import buffered_rpc_read, buffered_rpc_write
from ._function_utils import FunctionInfo
from .config import config
from .exception import ExecutionError, InvalidError, NotFoundError, RemoteError
from .mount import _create_client_mount, _Mount
from .object import Object
from .rate_limit import RateLimit
from .schedule import Schedule
from .secret import Secret

MODAL_CLIENT_MOUNT_NAME = "modal-client-mount"


async def _process_result(app, result):
    if result.WhichOneof("data_oneof") == "data_blob_id":
        data = await blob_download(result.data_blob_id, app.client.stub)
    else:
        data = result.data

    if result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
        if data:
            try:
                exc = app._deserialize(data)
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

            raise exc
        raise RemoteError(result.exception)

    return app._deserialize(data)


async def _create_input(args, kwargs, app, function_call_id, idx=None) -> api_pb2.FunctionInput:
    """Serialize function arguments and create a FunctionInput protobuf,
    uploading to blob storage if needed.
    """

    args_serialized = app._serialize((args, kwargs))

    if len(args_serialized) > MAX_OBJECT_SIZE_BYTES:
        args_blob_id = await blob_upload(args_serialized, app.client.stub)

        return api_pb2.FunctionInput(
            args_blob_id=args_blob_id,
            function_call_id=function_call_id,
            idx=idx,
        )
    else:
        return api_pb2.FunctionInput(
            args=args_serialized,
            function_call_id=function_call_id,
            idx=idx,
        )


class _Invocation:
    def __init__(self, app, function_id, function_call_id):
        self.app = app
        self.function_id = function_id
        self.function_call_id = function_call_id

    @staticmethod
    async def create(function_id, args, kwargs, app):
        if not function_id:
            raise InvalidError(
                "The function has not been initialized.\n"
                "\n"
                "Modal functions can only be called within an app. "
                "Try calling it from another running modal function or from an app run context:\n\n"
                "with app.run():\n"
                "    my_modal_function()\n"
            )
        request = api_pb2.FunctionMapRequest(function_id=function_id)
        response = await retry(app.client.stub.FunctionMap)(request)

        function_call_id = response.function_call_id

        inp = await _create_input(args, kwargs, app, function_call_id)
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
        return await _process_result(self.app, result)

    async def run_generator(self):
        completed = False
        while not completed:
            async for result in self.get_items():
                if result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                    completed = True
                    break
                yield await _process_result(self.app, result)


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

        input_queue: asyncio.Queue = asyncio.Queue()

        async def drain_input_generator():
            nonlocal num_inputs, input_queue
            async with self.input_stream.stream() as streamer:
                async for arg in streamer:
                    function_input = await _create_input(arg, self.kwargs, self.app, function_call_id, idx=num_inputs)
                    num_inputs += 1
                    await input_queue.put(function_input)
            # close queue iterator
            await input_queue.put(None)
            yield

        async def pump_inputs():
            nonlocal num_inputs, have_all_inputs, input_queue

            async for inputs in queue_batch_iterator(input_queue, MAP_INVOCATION_CHUNK_SIZE):
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
                            output = await _process_result(self.app, result)
                            # yield output directly for generators.
                            yield output
                    else:
                        # hold on to outputs for function maps, so we can reorder them correctly.
                        pending_outputs[result.idx] = await _process_result(self.app, result)

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

        response_gen = stream.merge(drain_input_generator(), pump_inputs(), poll_outputs())

        async with response_gen.stream() as streamer:
            async for response in streamer:
                # Handle yield at the end of pump_inputs, in case
                # that finishes after all outputs have been polled.
                if response is None:
                    if have_all_inputs and num_outputs == num_inputs:
                        break
                    continue
                yield response


class _Function(Object, type_prefix="fu"):
    def __init__(
        self,
        app,
        raw_f,
        image=None,
        secret: Optional[Secret] = None,
        secrets: Collection[Secret] = (),
        schedule: Optional[Schedule] = None,
        is_generator=False,
        gpu: bool = False,
        rate_limit: Optional[RateLimit] = None,
        # TODO: maybe break this out into a separate decorator for notebooks.
        serialized: bool = False,
        mounts: Collection[_Mount] = (),
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
    ):
        assert callable(raw_f)
        self.info = FunctionInfo(raw_f, serialized)
        if schedule is not None:
            if not self.info.is_nullary():
                raise InvalidError(
                    f"Function {raw_f} has a schedule, so it needs to support calling it with no arguments"
                )
        # assert not synchronizer.is_synchronized(image)

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
        self.mounts = mounts
        self.webhook_config = webhook_config
        self.web_url = None
        self.tag = self.info.get_tag()
        Object.__init__(self, app)

    def get_creating_message(self) -> str:
        return f"Creating {self.tag}..."

    def get_created_message(self) -> str:
        if self.web_url is not None:
            # TODO: this is only printed when we're showing progress. Maybe move this somewhere else.
            return f"Created {self.tag} => [magenta underline]{self.web_url}[/magenta underline]"
        return f"Created {self.tag}."

    async def load(self, app, existing_function_id):
        mounts = [*self.info.create_mounts(app), *self.mounts]
        # TODO(erikbern): couldn't we just create one single mount with all packages instead of multiple?
        if config["sync_entrypoint"]:
            mounts.append(await _create_client_mount(app))
        else:
            client_mount = _Mount.include(app, MODAL_CLIENT_MOUNT_NAME, namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)
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
            webhook_config=self.webhook_config,
        )
        request = api_pb2.FunctionCreateRequest(
            app_id=app.app_id,
            function=function_definition,
            schedule=self.schedule.proto_message if self.schedule is not None else None,
            existing_function_id=existing_function_id,
            deployment_name=self.app.deployment_name,
        )
        response = await app.client.stub.FunctionCreate(request)

        if response.web_url:
            self.web_url = response.web_url

        return response.function_id

    async def map(self, inputs, window, kwargs, is_generator):
        input_stream = stream.iterate(inputs) | pipe.map(lambda arg: (arg,))
        async for item in _MapInvocation(self.object_id, input_stream, kwargs, self._app, is_generator):
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


class _FunctionProxy:
    """This class represents a decorated function."""

    def __init__(self, orig_function, app, tag):
        # Need to store a reference to the unitialized function that has the contructor args
        self._orig_function = orig_function
        self._app = app
        self._tag = tag

    def _get_function(self):
        return self._app[self._tag]

    async def map(self, inputs, window=100, kwargs={}):
        async for it in self._get_function().map(inputs, window, kwargs, self._orig_function.is_generator):
            yield it

    def __call__(self, *args, **kwargs):
        if self._orig_function.is_generator:
            return self._get_function().call_generator(args, kwargs)
        else:
            return self._get_function().call_function(args, kwargs)

    async def enqueue(self, *args, **kwargs):
        """Calls the function with the given arguments without waiting for the results"""
        if self._orig_function.is_generator:
            await self._get_function().call_generator_nowait(args, kwargs)
        else:
            await self._get_function().call_function_nowait(args, kwargs)

    def get_raw_f(self):
        """Use by the container to get the code for the function."""
        return self._orig_function.raw_f

    @property
    def object_id(self):
        return self._get_function().object_id


# Note that we rename these
Function, AioFunction = synchronize_apis(_FunctionProxy)
