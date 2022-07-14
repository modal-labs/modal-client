import asyncio
import platform
from pathlib import Path
from typing import Collection, Dict, Optional, Union

from aiostream import stream

from modal_proto import api_pb2
from modal_utils.async_utils import queue_batch_iterator, synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors

from ._blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from ._buffer_utils import buffered_rpc_read
from ._function_utils import FunctionInfo
from ._serialization import deserialize, serialize
from .exception import ExecutionError, InvalidError, NotFoundError, RemoteError
from .mount import _Mount
from .object import Object, Ref
from .rate_limit import RateLimit
from .schedule import Schedule
from .secret import _Secret
from .shared_volume import _SharedVolume

MIN_MEMORY_MB = 1024
MAX_MEMORY_MB = 16 * 1024


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
        response = await retry_transient_errors(client.stub.FunctionMap, request)

        function_call_id = response.function_call_id

        inp = await _create_input(args, kwargs, client, function_call_id)
        request_put = api_pb2.FunctionPutInputsRequest(function_id=function_id, inputs=[inp])
        await retry_transient_errors(client.stub.FunctionPutInputs, request_put, max_retries=None)

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
        response = await retry_transient_errors(self.client.stub.FunctionMap, request)

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
                await retry_transient_errors(self.client.stub.FunctionPutInputs, request, max_retries=None)

            have_all_inputs = True
            yield

        async def poll_outputs():
            nonlocal num_inputs, num_outputs, have_all_inputs

            # map to store out-of-order outputs received
            pending_outputs = {}

            while True:
                request = api_pb2.FunctionGetOutputsRequest(function_call_id=function_call_id)
                response = await buffered_rpc_read(self.client.stub.FunctionGetOutputs, request, timeout=None)

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
    _secrets: Collection[Union[Ref, _Secret]]

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
        shared_volumes: Dict[str, Union[_SharedVolume, Ref]] = {},
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
        memory: Optional[int] = None,
        proxy: Optional[Ref] = None,
    ):
        assert callable(raw_f)
        self._info = FunctionInfo(raw_f, serialized)
        if schedule is not None:
            if not self._info.is_nullary():
                raise InvalidError(
                    f"Function {raw_f} has a schedule, so it needs to support calling it with no arguments"
                )
        # assert not synchronizer.is_synchronized(image)

        self._raw_f = raw_f
        self._image = image
        if secret and secrets:
            raise InvalidError(f"Function {raw_f} has both singular `secret` and plural `secrets` attached")
        if secret:
            self._secrets = [secret]
        else:
            self._secrets = secrets

        if memory is not None and memory < MIN_MEMORY_MB:
            raise InvalidError(f"Function {raw_f} memory request must be at least {MIN_MEMORY_MB} MB")
        elif memory is not None and memory >= MAX_MEMORY_MB:
            raise InvalidError(f"Function {raw_f} memory request must be less than {MAX_MEMORY_MB} MB")

        self._schedule = schedule
        self._is_generator = is_generator
        self._gpu = gpu
        self._rate_limit = rate_limit
        self._mounts = mounts
        self._shared_volumes = shared_volumes
        self._webhook_config = webhook_config
        self._web_url = None
        self._memory = memory
        self._proxy = proxy
        self._local_app = None
        self._local_object_id = None
        self._tag = self._info.get_tag()
        super().__init__()

    def _get_creating_message(self) -> str:
        return f"Creating {self._tag}..."

    def _get_created_message(self) -> str:
        if self._web_url is not None:
            # TODO: this is only printed when we're showing progress. Maybe move this somewhere else.
            return f"Created {self._tag} => [magenta underline]{self._web_url}[/magenta underline]"
        return f"Created {self._tag}."

    async def _load(self, client, app_id, existing_function_id):
        # TODO: should we really join recursively here? Maybe it's better to move this logic to the app class?
        if self._image is not None:
            image_id = await self._image
        else:
            image_id = None  # Happens if it's a notebook function
        secret_ids = []
        for secret in self._secrets:
            try:
                secret_id = await secret
            except NotFoundError as ex:
                raise NotFoundError(str(ex) + "\n" + "You can add secrets to your account at https://modal.com/secrets")
            secret_ids.append(secret_id)

        mount_ids = []
        for mount in self._mounts:
            mount_ids.append(await mount)

        if not isinstance(self._shared_volumes, dict):
            raise InvalidError("shared_volumes must be a dict[str, SharedVolume] where the keys are paths")
        shared_volume_mounts = []
        # Relies on dicts being ordered (true as of Python 3.6).
        for path, shared_volume in self._shared_volumes.items():
            # TODO: check paths client-side on Windows as well.
            if platform.system() != "Windows" and Path(path).resolve() != Path(path):
                raise InvalidError("Shared volume remote directory must be an absolute path.")

            shared_volume_mounts.append(
                api_pb2.SharedVolumeMount(mount_path=path, shared_volume_id=await shared_volume)
            )

        if self._is_generator:
            function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
        else:
            function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

        rate_limit = self._rate_limit.to_proto() if self._rate_limit else None

        if self._proxy:
            proxy_id = await self._proxy
        else:
            proxy_id = None

        # Create function remotely
        function_definition = api_pb2.Function(
            module_name=self._info.module_name,
            function_name=self._info.function_name,
            mount_ids=mount_ids,
            secret_ids=secret_ids,
            image_id=image_id,
            definition_type=self._info.definition_type,
            function_serialized=self._info.function_serialized,
            function_type=function_type,
            resources=api_pb2.Resources(gpu=self._gpu, memory=self._memory),
            rate_limit=rate_limit,
            webhook_config=self._webhook_config,
            shared_volume_mounts=shared_volume_mounts,
            proxy_id=proxy_id,
        )
        request = api_pb2.FunctionCreateRequest(
            app_id=app_id,
            function=function_definition,
            schedule=self._schedule.proto_message if self._schedule is not None else None,
            existing_function_id=existing_function_id,
        )
        response = await client.stub.FunctionCreate(request)

        if response.web_url:
            # TODO(erikbern): we really shouldn't mutate the object here
            self._web_url = response.web_url

        return response.function_id

    @property
    def tag(self):
        return self._tag

    @property
    def web_url(self):
        # TODO(erikbern): it would be much better if this gets written to the "live" object,
        # and then we look it up from the app.
        return self._web_url

    def set_local_app(self, app):
        """mdmd:hidden"""
        self._local_app = app

    def _get_context(self):
        # Functions are sort of "special" in the sense that they are just global objects not attached to an app
        # the way other objects are. So in order to work with functions, we need to look up the running app
        # in runtime. Either we're inside a container, in which case it's a singleton, or we're in the client,
        # in which case we can set the running app on all functions when we run the app.
        if self._client and self._object_id:
            # Can happen if this is a function loaded from a different app or something
            return self._object_id

        # avoid circular import
        from .app import _container_app, is_local

        if is_local():
            app = self._local_app
        else:
            app = _container_app
        client = app.client
        object_id = app[self._tag].object_id
        return (client, object_id)

    async def _map(self, input_stream, kwargs={}):
        client, object_id = self._get_context()
        async for item in _MapInvocation(object_id, input_stream, kwargs, client, self._is_generator):
            yield item

    async def map(
        self,
        *input_iterators,  # one input iterator per argument in the mapped-over function/generator
        kwargs={},  # any extra keyword arguments for the function
    ):
        """Parallel map over a set of inputs

        Takes one iterator argument per argument in the function being mapped over.

        Example:
        ```python notest
        @stub.function
        def my_func(a):
            return a ** 2

        assert list(my_func.starmap([1, 2, 3, 4])) == [1, 4, 9, 16]
        ```

        If applied to a `stub.function`, `map()` returns one result per input and the output order
        is guaranteed to be the same as the input order.

        If applied to a `stub.generator`, the results are returned as they are finished and can be
        out of order. By yielding zero or more than once, mapping over generators can also be used
        as a "flat map".
        """
        input_stream = stream.zip(*(stream.iterate(it) for it in input_iterators))
        async for item in self._map(input_stream, kwargs):
            yield item

    async def starmap(self, input_iterator, kwargs={}):
        """Like `map` but spreads arguments over multiple function arguments

        Assumes every input is a sequence (e.g. a tuple)

        Example:
        ```python notest
        @stub.function
        def my_func(a, b):
            return a + b

        assert list(my_func.starmap([(1, 2), (3, 4)])) == [3, 7]
        ```
        """
        input_stream = stream.iterate(input_iterator)
        async for item in self._map(input_stream, kwargs):
            yield item

    async def call_function(self, args, kwargs):
        """mdmd:hidden"""
        client, object_id = self._get_context()
        invocation = await Invocation.create(object_id, args, kwargs, client)
        return await invocation.run_function()

    async def call_function_nowait(self, args, kwargs):
        """mdmd:hidden"""
        client, object_id = self._get_context()
        await Invocation.create(object_id, args, kwargs, client)

    async def call_generator(self, args, kwargs):
        """mdmd:hidden"""
        client, object_id = self._get_context()
        invocation = await Invocation.create(object_id, args, kwargs, client)
        async for res in invocation.run_generator():
            yield res

    async def call_generator_nowait(self, args, kwargs):
        """mdmd:hidden"""
        client, object_id = self._get_context()
        await Invocation.create(object_id, args, kwargs, client)

    def __call__(self, *args, **kwargs):
        if self._is_generator:
            return self.call_generator(args, kwargs)
        else:
            return self.call_function(args, kwargs)

    async def enqueue(self, *args, **kwargs):
        """Calls the function with the given arguments without waiting for the results"""
        if self._is_generator:
            await self.call_generator_nowait(args, kwargs)
        else:
            await self.call_function_nowait(args, kwargs)

    def get_raw_f(self):
        """Use by the container to get the code for the function."""
        return self._raw_f


Function, AioFunction = synchronize_apis(_Function)
