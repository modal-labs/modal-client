import asyncio
import platform
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Collection,
    Dict,
    Optional,
    Tuple,
    Union,
)

from aiostream import pipe, stream
from grpclib import GRPCError, Status
from synchronicity.exceptions import UserCodeException

from modal_proto import api_pb2
from modal_utils.async_utils import (
    queue_batch_iterator,
    synchronize_apis,
    warn_if_generator_is_not_consumed,
)
from modal_utils.grpc_utils import retry_transient_errors

from ._blob_utils import (
    BLOB_MAX_PARALLELISM,
    MAX_OBJECT_SIZE_BYTES,
    blob_download,
    blob_upload,
)
from ._function_utils import FunctionInfo
from ._output import OutputManager
from ._serialization import deserialize, serialize
from ._traceback import append_modal_tb
from .client import _Client
from .exception import (
    ExecutionError,
    InvalidError,
    NotFoundError,
    RemoteError,
    deprecation_warning,
)
from .mount import _Mount
from .object import Handle, Provider, Ref, RemoteRef
from .rate_limit import RateLimit
from .retries import Retries
from .schedule import Schedule
from .secret import _Secret
from .shared_volume import _SharedVolume


def exc_with_hints(exc: BaseException):
    if isinstance(exc, ImportError) and exc.msg == "attempted relative import with no known parent package":
        exc.msg += """\n
HINT: For relative imports to work, you might need to run your modal app as a module. Try:
- `python -m my_pkg.my_app` instead of `python my_pkg/my_app.py`
- `modal app deploy my_pkg.my_app` instead of `modal app deploy my_pkg/my_app.py`
"""
    return exc


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

            if result.serialized_tb:
                try:
                    tb_dict = deserialize(result.serialized_tb, client)
                    line_cache = deserialize(result.tb_line_cache, client)
                    append_modal_tb(exc, tb_dict, line_cache)
                except Exception:
                    pass
            uc_exc = UserCodeException(exc_with_hints(exc))
            raise uc_exc
        raise RemoteError(result.exception)

    return deserialize(data, client)


async def _create_input(args, kwargs, client, idx=None) -> api_pb2.FunctionPutInputsItem:
    """Serialize function arguments and create a FunctionInput protobuf,
    uploading to blob storage if needed.
    """

    args_serialized = serialize((args, kwargs))

    if len(args_serialized) > MAX_OBJECT_SIZE_BYTES:
        args_blob_id = await blob_upload(args_serialized, client.stub)

        return api_pb2.FunctionPutInputsItem(
            input=api_pb2.FunctionInput(args_blob_id=args_blob_id),
            idx=idx,
        )
    else:
        return api_pb2.FunctionPutInputsItem(
            input=api_pb2.FunctionInput(args=args_serialized),
            idx=idx,
        )


@dataclass
class _OutputValue:
    # box class for distinguishing None results from non-existing/None markers
    value: Any


class _Invocation:
    def __init__(self, stub, function_call_id, client=None):
        self.stub = stub
        self.client = client  # Used by the deserializer.
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

        item = await _create_input(args, kwargs, client)
        request_put = api_pb2.FunctionPutInputsRequest(
            function_id=function_id, inputs=[item], function_call_id=function_call_id
        )
        await retry_transient_errors(
            client.stub.FunctionPutInputs,
            request_put,
            max_retries=None,
            additional_status_codes=[Status.RESOURCE_EXHAUSTED],
            ignore_errors=[Status.RESOURCE_EXHAUSTED],
        )

        return _Invocation(client.stub, function_call_id, client)

    async def get_items(self, timeout: float = None):
        t0 = time.time()
        if timeout is None:
            backend_timeout = 60.0
        else:
            backend_timeout = min(60.0, timeout)  # refresh backend call every 60s

        while True:
            # always execute at least one poll for results, regardless if timeout is 0
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=self.function_call_id, timeout=backend_timeout, return_empty_on_timeout=True
            )
            response = await retry_transient_errors(
                self.stub.FunctionGetOutputs,
                request,
            )
            if len(response.outputs) > 0:
                for item in response.outputs:
                    yield item.result
                return

            if timeout is not None:
                # update timeout in retry loop
                backend_timeout = min(60.0, t0 + timeout - time.time())
                if backend_timeout < 0:
                    break

    async def run_function(self):
        result = (await stream.list(self.get_items()))[0]
        assert not result.gen_status
        return await _process_result(result, self.stub, self.client)

    async def poll_function(self, timeout: float = 0):
        results = await stream.list(self.get_items(timeout=timeout))

        if len(results) == 0:
            raise TimeoutError()

        return await _process_result(results[0], self.stub, self.client)

    async def run_generator(self):
        completed = False
        while not completed:
            async for result in self.get_items():
                if result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                    completed = True
                    break
                yield await _process_result(result, self.stub, self.client)


MAP_INVOCATION_CHUNK_SIZE = 100


async def _map_invocation(
    function_id: str,
    input_stream: AsyncIterable,
    kwargs: Dict[str, Any],
    client: _Client,
    is_generator: bool,
    order_outputs: bool,
    count_update_callback: Optional[Callable[[int, int], None]],
):
    request = api_pb2.FunctionMapRequest(function_id=function_id)
    response = await retry_transient_errors(client.stub.FunctionMap, request)

    function_call_id = response.function_call_id

    have_all_inputs = False
    num_inputs = 0
    num_outputs = 0

    input_queue: asyncio.Queue = asyncio.Queue()

    async def create_input(arg):
        nonlocal num_inputs
        idx = num_inputs
        num_inputs += 1
        item = await _create_input(arg, kwargs, client, idx=idx)
        return item

    async def drain_input_generator():
        # Parallelize uploading blobs
        proto_input_stream = input_stream | pipe.map(create_input, ordered=True, task_limit=BLOB_MAX_PARALLELISM)
        async with proto_input_stream.stream() as streamer:
            async for item in streamer:
                await input_queue.put(item)

        # close queue iterator
        await input_queue.put(None)
        yield

    async def pump_inputs():
        nonlocal have_all_inputs
        async for items in queue_batch_iterator(input_queue, MAP_INVOCATION_CHUNK_SIZE):
            request = api_pb2.FunctionPutInputsRequest(
                function_id=function_id, inputs=items, function_call_id=function_call_id
            )
            await retry_transient_errors(
                client.stub.FunctionPutInputs,
                request,
                max_retries=None,
                additional_status_codes=[Status.RESOURCE_EXHAUSTED],
                ignore_errors=[Status.RESOURCE_EXHAUSTED],
            )

        have_all_inputs = True
        yield

    async def get_all_outputs():
        nonlocal num_inputs, num_outputs, have_all_inputs
        while not have_all_inputs or num_outputs < num_inputs:
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=function_call_id, timeout=60, return_empty_on_timeout=True
            )
            response = await retry_transient_errors(
                client.stub.FunctionGetOutputs,
                request,
                max_retries=None,
                base_delay=0,
            )
            for item in response.outputs:
                if is_generator:
                    if item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                        num_outputs += 1
                    else:
                        yield item
                else:
                    num_outputs += 1
                    yield item

    async def fetch_output(item):
        output = await _process_result(item.result, client.stub, client)
        return (item.idx, output)

    async def poll_outputs():
        outputs = stream.iterate(get_all_outputs())
        outputs_fetched = outputs | pipe.map(fetch_output, ordered=True, task_limit=BLOB_MAX_PARALLELISM)

        # map to store out-of-order outputs received
        pending_outputs = {}
        output_idx = 0

        async with outputs_fetched.stream() as streamer:
            async for idx, output in streamer:
                if count_update_callback is not None:
                    count_update_callback(num_outputs, num_inputs)
                if is_generator:
                    yield _OutputValue(output)
                elif not order_outputs:
                    yield _OutputValue(output)
                else:
                    # hold on to outputs for function maps, so we can reorder them correctly.
                    pending_outputs[idx] = output
                    while output_idx in pending_outputs:
                        output = pending_outputs.pop(output_idx)
                        yield _OutputValue(output)
                        output_idx += 1

        assert len(pending_outputs) == 0

    response_gen = stream.merge(drain_input_generator(), pump_inputs(), poll_outputs())

    async with response_gen.stream() as streamer:
        async for response in streamer:
            if response is not None:
                yield response.value


class _FunctionHandle(Handle, type_prefix="fu"):
    """Interact with a Modal Function of a live app."""

    def __init__(self, function, web_url=None, client=None, object_id=None):
        self._local_app = None
        self._progress = None

        # These are some stupid lines, let's rethink
        self._tag = function._tag
        self._is_generator = function._is_generator
        self._raw_f = function._raw_f
        self._web_url = web_url
        self._output_mgr: Optional[OutputManager] = None
        self._mute_cancellation = (
            False  # set when a user terminates the app intentionally, to prevent useless traceback spam
        )

        super().__init__(client=client, object_id=object_id)

    def _set_mute_cancellation(self, value=True):
        self._mute_cancellation = value

    def _initialize_from_proto(self, function: api_pb2.Function):
        self._is_generator = function.function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR

    def _set_local_app(self, app):
        """mdmd:hidden"""
        self._local_app = app

    def _set_output_mgr(self, output_mgr: OutputManager):
        """mdmd:hidden"""
        self._output_mgr = output_mgr

    def _get_live_handle(self) -> "_FunctionHandle":
        # Functions are sort of "special" in the sense that they are just global objects not attached to an app
        # the way other objects are. So in order to work with functions, we need to look up the running app
        # in runtime. Either we're inside a container, in which case it's a singleton, or we're in the client,
        # in which case we can set the running app on all functions when we run the app.
        if self._client and self._object_id:
            # Can happen if this is a function loaded from a different app or something
            return self

        # avoid circular import
        from .app import _container_app, is_local

        if is_local():
            if self._local_app is None:
                raise InvalidError(
                    "App is not running. You might need to put the function call inside a `with stub.run():` block."
                )
            app = self._local_app
        else:
            app = _container_app
        obj = app[self._tag]
        assert isinstance(obj, _FunctionHandle)
        return obj

    def _get_context(self) -> Tuple[_Client, str]:
        function_handle = self._get_live_handle()
        return (function_handle._client, function_handle._object_id)

    @property
    def web_url(self) -> str:
        """URL of a Function running as a web endpoint."""
        function_handle = self._get_live_handle()
        return function_handle._web_url

    async def _map(self, input_stream: AsyncIterable, order_outputs: bool, kwargs={}):
        if order_outputs and self._is_generator:
            raise ValueError("Can't return ordered results for a generator")

        client, object_id = self._get_context()

        count_update_callback = self._output_mgr.function_progress_callback(self._tag) if self._output_mgr else None

        async for item in _map_invocation(
            object_id, input_stream, kwargs, client, self._is_generator, order_outputs, count_update_callback
        ):
            yield item

    @warn_if_generator_is_not_consumed
    async def map(
        self,
        *input_iterators,  # one input iterator per argument in the mapped-over function/generator
        kwargs={},  # any extra keyword arguments for the function
        order_outputs=None,  # defaults to True for regular functions, False for generators
    ):
        """Parallel map over a set of inputs.

        Takes one iterator argument per argument in the function being mapped over.

        Example:
        ```python notest
        @stub.function
        def my_func(a):
            return a ** 2

        assert list(my_func.map([1, 2, 3, 4])) == [1, 4, 9, 16]
        ```

        If applied to a `stub.function`, `map()` returns one result per input and the output order
        is guaranteed to be the same as the input order. Set `order_output=False` to return results
        in the order that they are completed instead.

        If applied to a `stub.generator`, the results are returned as they are finished and can be
        out of order. By yielding zero or more than once, mapping over generators can also be used
        as a "flat map".
        """
        if order_outputs is None:
            order_outputs = not self._is_generator

        input_stream = stream.zip(*(stream.iterate(it) for it in input_iterators))
        async for item in self._map(input_stream, order_outputs, kwargs):
            yield item

    async def for_each(self, *input_iterators, **kwargs):
        """Execute function for all outputs, ignoring outputs

        Convenient alias for `.map()` in cases where the function just needs to be called.
        as the caller doesn't have to consume the generator to process the inputs.
        """
        async for _ in self.map(*input_iterators, order_outputs=False, **kwargs):
            pass

    @warn_if_generator_is_not_consumed
    async def starmap(self, input_iterator, kwargs={}, order_outputs=None):
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
        if order_outputs is None:
            order_outputs = not self._is_generator

        input_stream = stream.iterate(input_iterator)
        async for item in self._map(input_stream, order_outputs, kwargs):
            yield item

    async def call_function(self, args, kwargs):
        """mdmd:hidden"""
        client, object_id = self._get_context()
        invocation = await _Invocation.create(object_id, args, kwargs, client)
        try:
            return await invocation.run_function()
        except asyncio.CancelledError:
            # this can happen if the user terminates a program, triggering a cancellation cascade
            if not self._mute_cancellation:
                raise

    async def call_function_nowait(self, args, kwargs):
        """mdmd:hidden"""
        client, object_id = self._get_context()
        return await _Invocation.create(object_id, args, kwargs, client)

    @warn_if_generator_is_not_consumed
    async def call_generator(self, args, kwargs):
        """mdmd:hidden"""
        client, object_id = self._get_context()
        invocation = await _Invocation.create(object_id, args, kwargs, client)
        async for res in invocation.run_generator():
            yield res

    async def call_generator_nowait(self, args, kwargs):
        """mdmd:hidden"""
        client, object_id = self._get_context()
        return await _Invocation.create(object_id, args, kwargs, client)

    def __call__(self, *args, **kwargs):
        if self._is_generator:
            return self.call_generator(args, kwargs)
        else:
            return self.call_function(args, kwargs)

    async def enqueue(self, *args, **kwargs):
        """**Deprecated.** Use `.submit()` instead when possible.

        Calls the function with the given arguments, without waiting for the results.
        """
        deprecation_warning("Function.enqueue is deprecated, use .submit() instead")
        if self._is_generator:
            await self.call_generator_nowait(args, kwargs)
        else:
            await self.call_function_nowait(args, kwargs)

    async def submit(self, *args, **kwargs) -> Optional["_FunctionCall"]:
        """Calls the function with the given arguments, without waiting for the results.

        Returns a `modal.functions.FunctionCall` object, that can later be polled or waited for using `.get(timeout=...)`.
        Conceptually similar to `multiprocessing.pool.apply_async`, or a Future/Promise in other contexts.

        *Note:* `.submit()` on a modal generator function does call and execute the generator, but does not currently
        return a function handle for polling the result.
        """
        if self._is_generator:
            await self.call_generator_nowait(args, kwargs)
            return None

        invocation = await self.call_function_nowait(args, kwargs)
        return _FunctionCall(invocation.client, invocation.function_call_id)

    def get_raw_f(self) -> Callable:
        """Return the inner Python object wrapped by this Modal Function."""
        return self._raw_f


FunctionHandle, AioFunctionHandle = synchronize_apis(_FunctionHandle)


class _Function(Provider[_FunctionHandle]):
    """Functions are the basic units of serverless execution on Modal.

    Generally, you will not construct a `Function` directly. Instead, use the
    `@stub.function` decorator on the `Stub` object for your application.
    """

    # TODO: more type annotations
    _secrets: Collection[_Secret]

    def __init__(
        self,
        raw_f,
        image=None,
        secrets: Collection[_Secret] = (),
        schedule: Optional[Schedule] = None,
        is_generator=False,
        gpu: bool = False,
        rate_limit: Optional[RateLimit] = None,
        # TODO: maybe break this out into a separate decorator for notebooks.
        serialized: bool = False,
        mounts: Collection[_Mount] = (),
        shared_volumes: Dict[str, _SharedVolume] = {},
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
        memory: Optional[int] = None,
        proxy: Optional[Ref] = None,
        retries: Optional[Union[int, Retries]] = None,
        concurrency_limit: Optional[int] = None,
        cpu: Optional[float] = None,
        keep_warm: bool = False,
    ) -> None:
        """mdmd:hidden"""
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
        self._secrets = secrets

        if retries:
            if isinstance(retries, int):
                retry_policy = Retries(
                    max_retries=retries,
                    initial_delay=1.0,
                    backoff_coefficient=1.0,
                )
            elif isinstance(retries, Retries):
                retry_policy = retries
            else:
                raise InvalidError(
                    f"Function {raw_f} retries must be an integer or instance of modal.Retries. Found: {type(retries)}"
                )

            if not (0 <= retry_policy.max_retries <= 10):
                raise InvalidError(f"Function {raw_f} retries must be between 0 and 10.")

            # TODO(Jonathon): Right now we can only support a maximum delay of 60 seconds
            # b/c the CONTAINER_IDLE_TIMEOUT is the maximum time a container will wait for inputs,
            # after which it will exit 0 and the task is marked finished.
            if not (timedelta(seconds=1) < retry_policy.max_delay <= timedelta(seconds=60)):
                raise InvalidError(
                    f"Invalid max_delay argument: {repr(retry_policy.max_delay)}. Must be between 1-60 seconds."
                )

            # initial_delay should be bounded by max_delay, but this is an extra defensive check.
            if not (timedelta(seconds=0) < retry_policy.initial_delay <= timedelta(seconds=60)):
                raise InvalidError(
                    f"Invalid initial_delay argument: {repr(retry_policy.initial_delay)}. Must be between 0-60 seconds."
                )
        else:
            retry_policy = None

        self._schedule = schedule
        self._is_generator = is_generator
        self._gpu = gpu
        self._rate_limit = rate_limit
        self._mounts = mounts
        self._shared_volumes = shared_volumes
        self._webhook_config = webhook_config
        self._cpu = cpu
        self._memory = memory
        self._proxy = proxy
        self._retry_policy = retry_policy
        self._concurrency_limit = concurrency_limit
        self._keep_warm = keep_warm
        self._tag = self._info.get_tag()
        super().__init__()

    async def _load(self, client, app_id, loader, message_callback, existing_function_id):
        message_callback(f"Creating {self._tag}...")

        if self._proxy:
            proxy_id = await loader(self._proxy)
            # HACK: remove this once we stop using ssh tunnels for this.
            if self._image:
                self._image = self._image.run_commands(["apt-get install -yq ssh"])
        else:
            proxy_id = None

        # TODO: should we really join recursively here? Maybe it's better to move this logic to the app class?
        if self._image is not None:
            image_id = await loader(self._image)
        else:
            image_id = None  # Happens if it's a notebook function
        secret_ids = []
        for secret in self._secrets:
            try:
                secret_id = await loader(secret)
            except NotFoundError as ex:
                if isinstance(secret, RemoteRef) and secret.tag is None:
                    msg = "Secret {!r} was not found".format(secret.app_name)
                else:
                    msg = str(ex)
                msg += ". You can add secrets to your account at https://modal.com/secrets"
                raise NotFoundError(msg)
            secret_ids.append(secret_id)

        mount_ids = []
        for mount in self._mounts:
            mount_ids.append(await loader(mount))

        if not isinstance(self._shared_volumes, dict):
            raise InvalidError("shared_volumes must be a dict[str, SharedVolume] where the keys are paths")
        shared_volume_mounts = []
        # Relies on dicts being ordered (true as of Python 3.6).
        for path, shared_volume in self._shared_volumes.items():
            # TODO: check paths client-side on Windows as well.
            if platform.system() != "Windows" and Path(path).resolve() != Path(path):
                raise InvalidError("Shared volume remote directory must be an absolute path.")

            shared_volume_mounts.append(
                api_pb2.SharedVolumeMount(mount_path=path, shared_volume_id=await loader(shared_volume))
            )

        if self._is_generator:
            function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
        else:
            function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

        rate_limit = self._rate_limit._to_proto() if self._rate_limit else None
        retry_policy = self._retry_policy._to_proto() if self._retry_policy else None

        if self._cpu is not None and self._cpu < 0.0:
            raise InvalidError(f"Invalid fractional CPU value {self._cpu}. Cannot have negative CPU resources.")
        milli_cpu = int(1000 * self._cpu) if self._cpu is not None else None

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
            resources=api_pb2.Resources(milli_cpu=milli_cpu, gpu=self._gpu, memory_mb=self._memory),
            rate_limit=rate_limit,
            webhook_config=self._webhook_config,
            shared_volume_mounts=shared_volume_mounts,
            proxy_id=proxy_id,
            retry_policy=retry_policy,
            concurrency_limit=self._concurrency_limit,
            keep_warm=self._keep_warm,
        )
        request = api_pb2.FunctionCreateRequest(
            app_id=app_id,
            function=function_definition,
            schedule=self._schedule.proto_message if self._schedule is not None else None,
            existing_function_id=existing_function_id,
        )
        try:
            response = await client.stub.FunctionCreate(request)
        except GRPCError as exc:
            if exc.status == Status.INVALID_ARGUMENT:
                raise InvalidError(exc.message)
            raise

        if response.web_url:
            # TODO: this is only printed when we're showing progress. Maybe move this somewhere else.
            message_callback(f"Created {self._tag} => [magenta underline]{response.web_url}[/magenta underline]")
        else:
            message_callback(f"Created {self._tag}.")

        return _FunctionHandle(self, response.web_url, client, response.function_id)

    @property
    def tag(self):
        """mdmd:hidden"""
        return self._tag


Function, AioFunction = synchronize_apis(_Function)


class _FunctionCall(Handle, type_prefix="fc"):
    """A reference to an executed function call

    Constructed using `.submit(...)` on a Modal function with the same
    arguments that a function normally takes. Acts as a reference to
    an ongoing function call that can be passed around and used to
    poll or fetch function results at some later time.

    Conceptually similar to a Future/Promise/AsyncResult in other contexts and languages.
    """

    def _invocation(self):
        return _Invocation(self._client.stub, self.object_id, self._client)

    async def get(self, timeout: Optional[float] = None):
        """Gets the result of the function call

        Raises `TimeoutError` if no results are returned within `timeout` seconds.
        Setting `timeout` to None (the default) waits indefinitely until there is a result
        """
        return await self._invocation().poll_function(timeout=timeout)


FunctionCall, AioFunctionCall = synchronize_apis(_FunctionCall)


async def _gather(*function_calls: _FunctionCall):
    """Wait until all Modal function calls have results before returning

    Accepts a variable number of FunctionCall objects as returned by `Function.submit()`.

    Returns a list of results from each function call, or raises an exception
    of the first failing function call.

    E.g.

    ```python notest
    function_call_1 = slow_func_1.submit()
    function_call_2 = slow_func_2.submit()

    result_1, result_2 = gather(function_call_1, function_call_2)
    ```
    """
    try:
        return await asyncio.gather(*[fc.get() for fc in function_calls])
    except Exception as exc:
        # TODO: kill all running function calls
        raise exc


gather, aio_gather = synchronize_apis(_gather)


_current_input_id: Optional[str] = None


def current_input_id() -> str:
    """Returns the input id for the currently processed input

    Can only be called from Modal function (i.e. in a container context)

    ```python
    from modal import current_input_id

    @stub.function
    def process_stuff():
        print(f"Starting to process {current_input_id()}")
    ```
    """
    global _current_input_id

    if _current_input_id is None:
        raise Exception("current_input_id is only available within a container-executed Modal function")

    return _current_input_id


def _set_current_input_id(input_id: str):
    global _current_input_id
    _current_input_id = input_id
