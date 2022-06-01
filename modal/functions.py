import asyncio
from typing import Collection, Optional, Union

from aiostream import pipe, stream

from modal_proto import api_pb2
from modal_utils.async_utils import queue_batch_iterator, retry, synchronize_apis

from ._blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from ._buffer_utils import buffered_rpc_read, buffered_rpc_write
from ._function_utils import FunctionInfo
from ._serialization import deserialize, serialize
from .exception import ExecutionError, InvalidError, NotFoundError, RemoteError
from .mount import _Mount
from .object import Object, Ref
from .rate_limit import RateLimit
from .schedule import Schedule
from .secret import _Secret


async def _process_result(result, stub, client=None):
    if result.WhichOneof("data_oneof") == "data_blob_id":
        data = await blob_download(result.data_blob_id, stub)
    else:
        data = result.data

    if result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
        if data:
            try:
                exc = deserialize(data, client)
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

    return deserialize(data, client)


async def _create_input(args, kwargs, client, function_call_id, idx=None) -> api_pb2.FunctionInput:
    """Serialize function arguments and create a FunctionInput protobuf,
    uploading to blob storage if needed.
    """

    args_serialized = serialize((args, kwargs))

    if len(args_serialized) > MAX_OBJECT_SIZE_BYTES:
        args_blob_id = await blob_upload(args_serialized, client.stub)

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


class Invocation:
    def __init__(self, stub, function_id, function_call_id, client=None):
        self.stub = stub
        self.client = client  # Used by the deserializer.
        self.function_id = function_id
        self.function_call_id = function_call_id

    @staticmethod
    async def create(function_id, args, kwargs, client):
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
        response = await retry(client.stub.FunctionMap)(request)

        function_call_id = response.function_call_id

        inp = await _create_input(args, kwargs, client, function_call_id)
        request_put = api_pb2.FunctionPutInputsRequest(function_id=function_id, inputs=[inp])
        await buffered_rpc_write(client.stub.FunctionPutInputs, request_put)

        return Invocation(client.stub, function_id, function_call_id, client)

    async def get_items(self):
        request = api_pb2.FunctionGetOutputsRequest(function_call_id=self.function_call_id)
        response = await buffered_rpc_read(self.stub.FunctionGetOutputs, request, timeout=None)
        for output in response.outputs:
            yield output

    async def run_function(self):
        result = (await stream.list(self.get_items()))[0]
        assert not result.gen_status
        return await _process_result(result, self.stub, self.client)

    async def run_generator(self):
        completed = False
        while not completed:
            async for result in self.get_items():
                if result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                    completed = True
                    break
                yield await _process_result(result, self.stub, self.client)


MAP_INVOCATION_CHUNK_SIZE = 100


class _MapInvocation:
    # TODO: should this be an object?
    def __init__(self, function_id, input_stream, kwargs, client, is_generator):
        self.function_id = function_id
        self.input_stream = input_stream
        self.kwargs = kwargs
        self.client = client
        self.is_generator = is_generator

    async def __aiter__(self):
        request = api_pb2.FunctionMapRequest(function_id=self.function_id)
        response = await retry(self.client.stub.FunctionMap)(request)

        function_call_id = response.function_call_id

        have_all_inputs = False
        num_outputs = 0
        num_inputs = 0

        input_queue: asyncio.Queue = asyncio.Queue()

        async def drain_input_generator():
            nonlocal num_inputs, input_queue
            async with self.input_stream.stream() as streamer:
                async for arg in streamer:
                    function_input = await _create_input(
                        arg, self.kwargs, self.client, function_call_id, idx=num_inputs
                    )
                    num_inputs += 1
                    await input_queue.put(function_input)
            # close queue iterator
            await input_queue.put(None)
            yield

        async def pump_inputs():
            nonlocal num_inputs, have_all_inputs, input_queue

            async for inputs in queue_batch_iterator(input_queue, MAP_INVOCATION_CHUNK_SIZE):
                request = api_pb2.FunctionPutInputsRequest(function_id=self.function_id, inputs=inputs)
                await buffered_rpc_write(self.client.stub.FunctionPutInputs, request)

            have_all_inputs = True
            yield

        async def poll_outputs():
            nonlocal num_inputs, num_outputs, have_all_inputs

            # map to store out-of-order outputs received
            pending_outputs = {}

            while True:
                request = api_pb2.FunctionGetOutputsRequest(function_call_id=function_call_id)
                response = await buffered_rpc_read(
                    self.client.stub.FunctionGetOutputs, request, timeout=None, warn_on_cancel=False
                )

                for result in response.outputs:
                    if self.is_generator:
                        if result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                            num_outputs += 1
                        else:
                            output = await _process_result(result, self.client.stub, self.client)
                            # yield output directly for generators.
                            yield output
                    else:
                        # hold on to outputs for function maps, so we can reorder them correctly.
                        pending_outputs[result.idx] = await _process_result(result, self.client.stub, self.client)

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
    # TODO: more type annotations
    secrets: Collection[Union[Ref, _Secret]]

    def __init__(
        self,
        raw_f,
        image=None,
        secret: Optional[Union[Ref, _Secret]] = None,
        secrets: Collection[Union[Ref, _Secret]] = (),
        schedule: Optional[Schedule] = None,
        is_generator=False,
        gpu: bool = False,
        rate_limit: Optional[RateLimit] = None,
        # TODO: maybe break this out into a separate decorator for notebooks.
        serialized: bool = False,
        mounts: Collection[Union[Ref, _Mount]] = (),
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
            self.secrets = [secret]
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
        self._local_running_app = None
        self._local_object_id = None
        super().__init__()

    def get_creating_message(self) -> str:
        return f"Creating {self.tag}..."

    def get_created_message(self) -> str:
        if self.web_url is not None:
            # TODO: this is only printed when we're showing progress. Maybe move this somewhere else.
            return f"Created {self.tag} => [magenta underline]{self.web_url}[/magenta underline]"
        return f"Created {self.tag}."

    async def load(self, client, app_id, existing_function_id):
        # TODO: should we really join recursively here? Maybe it's better to move this logic to the app class?
        if self.image is not None:
            image_id = await self.image
        else:
            image_id = None  # Happens if it's a notebook function
        secret_ids = []
        for secret in self.secrets:
            try:
                secret_id = await secret
            except NotFoundError as ex:
                raise NotFoundError(str(ex) + "\n" + "You can add secrets to your account at https://modal.com/secrets")
            secret_ids.append(secret_id)

        mount_ids = []
        for mount in self.mounts:
            mount_ids.append(await mount)

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
            app_id=app_id,
            function=function_definition,
            schedule=self.schedule.proto_message if self.schedule is not None else None,
            existing_function_id=existing_function_id,
        )
        response = await client.stub.FunctionCreate(request)

        if response.web_url:
            self.web_url = response.web_url

        return response.function_id

    def set_local_running_app(self, running_app):
        self._local_running_app = running_app

    def _get_context(self):
        # Functions are sort of "special" in the sense that they are just global objects not attached to an app
        # the way other objects are. So in order to work with functions, we need to look up the running app
        # in runtime. Either we're inside a container, in which case it's a singleton, or we're in the client,
        # in which case we can set the running app on all functions when we run the app.
        if self._client and self._object_id:
            # Can happen if this is a function loaded from a different app or something
            return self._object_id

        from .app import _container_app, is_local  # avoid circular import

        if is_local():
            running_app = self._local_running_app
        else:
            running_app = _container_app
        client = running_app.client
        object_id = running_app[self.tag].object_id
        return (client, object_id)

    async def map(self, inputs, window=100, kwargs={}):
        client, object_id = self._get_context()
        input_stream = stream.iterate(inputs) | pipe.map(lambda arg: (arg,))
        async for item in _MapInvocation(object_id, input_stream, kwargs, client, self.is_generator):
            yield item

    async def call_function(self, args, kwargs):
        client, object_id = self._get_context()
        invocation = await Invocation.create(object_id, args, kwargs, client)
        return await invocation.run_function()

    async def call_function_nowait(self, args, kwargs):
        client, object_id = self._get_context()
        await Invocation.create(object_id, args, kwargs, client)

    async def call_generator(self, args, kwargs):
        client, object_id = self._get_context()
        invocation = await Invocation.create(object_id, args, kwargs, client)
        async for res in invocation.run_generator():
            yield res

    async def call_generator_nowait(self, args, kwargs):
        client, object_id = self._get_context()
        await Invocation.create(object_id, args, kwargs, client)

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
        """Use by the container to get the code for the function."""
        return self.raw_f


Function, AioFunction = synchronize_apis(_Function)
