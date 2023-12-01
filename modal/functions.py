# Copyright Modal Labs 2022
import asyncio
import inspect
import os
import pickle
import time
import typing
import warnings
from dataclasses import dataclass
from datetime import date
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Union,
)

from aiostream import pipe, stream
from google.protobuf.message import Message
from grpclib import GRPCError, Status
from synchronicity.exceptions import UserCodeException

from modal import _pty
from modal._types import typechecked
from modal_proto import api_pb2
from modal_utils.async_utils import (
    queue_batch_iterator,
    synchronize_api,
    synchronizer,
    warn_if_generator_is_not_consumed,
)
from modal_utils.grpc_utils import retry_transient_errors

from ._blob_utils import (
    BLOB_MAX_PARALLELISM,
    MAX_OBJECT_SIZE_BYTES,
    blob_download,
    blob_upload,
)
from ._function_utils import FunctionInfo, is_async
from ._location import parse_cloud_provider
from ._mount_utils import validate_mount_points
from ._output import OutputManager
from ._resolver import Resolver
from ._serialization import deserialize, deserialize_data_format, serialize
from ._traceback import append_modal_tb
from .call_graph import InputInfo, _reconstruct_call_graph
from .client import _Client
from .config import config, logger
from .exception import (
    ExecutionError,
    FunctionTimeoutError,
    InvalidError,
    RemoteError,
    deprecation_error,
)
from .gpu import GPU_T, parse_gpu_config
from .image import _Image
from .mount import _get_client_mount, _Mount
from .network_file_system import _NetworkFileSystem, network_file_system_mount_protos
from .object import _Object, live_method, live_method_gen
from .proxy import _Proxy
from .retries import Retries
from .schedule import Schedule
from .secret import _Secret
from .volume import _Volume

ATTEMPT_TIMEOUT_GRACE_PERIOD = 5  # seconds


if typing.TYPE_CHECKING:
    import modal.stub


def exc_with_hints(exc: BaseException):
    """mdmd:hidden"""
    if isinstance(exc, ImportError) and exc.msg == "attempted relative import with no known parent package":
        exc.msg += """\n
HINT: For relative imports to work, you might need to run your modal app as a module. Try:
- `python -m my_pkg.my_app` instead of `python my_pkg/my_app.py`
- `modal deploy my_pkg.my_app` instead of `modal deploy my_pkg/my_app.py`
"""
    elif isinstance(
        exc, RuntimeError
    ) and "CUDA error: no kernel image is available for execution on the device" in str(exc):
        msg = (
            exc.args[0]
            + """\n
HINT: This error usually indicates an outdated CUDA version. Older versions of torch (<=1.12)
come with CUDA 10.2 by default. If pinning to an older torch version, you can specify a CUDA version
manually, for example:
-  image.pip_install("torch==1.12.1+cu116", find_links="https://download.pytorch.org/whl/torch_stable.html")
"""
        )
        exc.args = (msg,)

    return exc


async def _process_result(result: api_pb2.GenericResult, data_format: int, stub, client=None):
    if result.WhichOneof("data_oneof") == "data_blob_id":
        data = await blob_download(result.data_blob_id, stub)
    else:
        data = result.data

    if result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
        raise FunctionTimeoutError(result.exception)
    elif result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
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

    try:
        return deserialize_data_format(data, data_format, client)
    except ModuleNotFoundError as deser_exc:
        raise ExecutionError(
            "Could not deserialize result due to error:\n"
            + f"{deser_exc}\n"
            + "This can happen if your local environment does not have a module that was used to construct the result. \n"
        )


async def _create_input(args, kwargs, client, idx=None) -> api_pb2.FunctionPutInputsItem:
    """Serialize function arguments and create a FunctionInput protobuf,
    uploading to blob storage if needed.
    """

    args_serialized = serialize((args, kwargs))

    if len(args_serialized) > MAX_OBJECT_SIZE_BYTES:
        args_blob_id = await blob_upload(args_serialized, client.stub)

        return api_pb2.FunctionPutInputsItem(
            input=api_pb2.FunctionInput(args_blob_id=args_blob_id, data_format=api_pb2.DATA_FORMAT_PICKLE),
            idx=idx,
        )
    else:
        return api_pb2.FunctionPutInputsItem(
            input=api_pb2.FunctionInput(args=args_serialized, data_format=api_pb2.DATA_FORMAT_PICKLE),
            idx=idx,
        )


@dataclass
class _OutputValue:
    # box class for distinguishing None results from non-existing/None markers
    value: Any


class _Invocation:
    """Internal client representation of a single-input call to a Modal Function or Generator"""

    def __init__(self, stub, function_call_id, client=None):
        self.stub = stub
        self.client = client  # Used by the deserializer.
        self.function_call_id = function_call_id  # TODO: remove and use only input_id

    @staticmethod
    async def create(function_id: str, args, kwargs, client: _Client):
        request = api_pb2.FunctionMapRequest(
            function_id=function_id,
            parent_input_id=current_input_id(),
            function_call_type=api_pb2.FUNCTION_CALL_TYPE_UNARY,
        )
        response = await retry_transient_errors(client.stub.FunctionMap, request)

        function_call_id = response.function_call_id

        item = await _create_input(args, kwargs, client)
        request_put = api_pb2.FunctionPutInputsRequest(
            function_id=function_id, inputs=[item], function_call_id=function_call_id
        )
        inputs_response: api_pb2.FunctionPutInputsResponse = await retry_transient_errors(
            client.stub.FunctionPutInputs,
            request_put,
        )
        processed_inputs = inputs_response.inputs
        if not processed_inputs:
            raise Exception("Could not create function call - the input queue seems to be full")
        return _Invocation(client.stub, function_call_id, client)

    async def pop_function_call_outputs(
        self, timeout: Optional[float], clear_on_success: bool
    ) -> AsyncIterator[api_pb2.FunctionGetOutputsItem]:
        t0 = time.time()
        if timeout is None:
            backend_timeout = config["outputs_timeout"]
        else:
            backend_timeout = min(config["outputs_timeout"], timeout)  # refresh backend call every 55s

        while True:
            # always execute at least one poll for results, regardless if timeout is 0
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=self.function_call_id,
                timeout=backend_timeout,
                last_entry_id="0-0",
                clear_on_success=clear_on_success,
            )
            response: api_pb2.FunctionGetOutputsResponse = await retry_transient_errors(
                self.stub.FunctionGetOutputs,
                request,
                attempt_timeout=backend_timeout + ATTEMPT_TIMEOUT_GRACE_PERIOD,
            )
            if len(response.outputs) > 0:
                for item in response.outputs:
                    yield item
                return

            if timeout is not None:
                # update timeout in retry loop
                backend_timeout = min(config["outputs_timeout"], t0 + timeout - time.time())
                if backend_timeout < 0:
                    break

    async def run_function(self):
        # waits indefinitely for a single result for the function, and clear the outputs buffer after
        item: api_pb2.FunctionGetOutputsItem = (
            await stream.list(self.pop_function_call_outputs(timeout=None, clear_on_success=True))
        )[0]
        assert not item.result.gen_status
        return await _process_result(item.result, item.data_format, self.stub, self.client)

    async def poll_function(self, timeout: Optional[float] = None):
        """Waits up to timeout for a result from a function.

        If timeout is `None`, waits indefinitely. This function is not
        cancellation-safe.
        """
        items: List[api_pb2.FunctionGetOutputsItem] = await stream.list(
            self.pop_function_call_outputs(timeout=timeout, clear_on_success=False)
        )

        if len(items) == 0:
            raise TimeoutError()

        return await _process_result(items[0].result, items[0].data_format, self.stub, self.client)

    async def run_generator(self):
        last_entry_id = "0-0"
        completed = False
        try:
            while not completed:
                request = api_pb2.FunctionGetOutputsRequest(
                    function_call_id=self.function_call_id,
                    timeout=config["outputs_timeout"],
                    last_entry_id=last_entry_id,
                    clear_on_success=False,  # there could be more results
                )
                response: api_pb2.FunctionGetOutputsResponse = await retry_transient_errors(
                    self.stub.FunctionGetOutputs,
                    request,
                    attempt_timeout=config["outputs_timeout"] + ATTEMPT_TIMEOUT_GRACE_PERIOD,
                )
                if len(response.outputs) > 0:
                    last_entry_id = response.last_entry_id
                    for item in response.outputs:
                        if item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                            completed = True
                            break
                        yield await _process_result(item.result, item.data_format, self.stub, self.client)
        finally:
            # "ack" that we have all outputs we are interested in and let backend clear results
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=self.function_call_id,
                timeout=0,
                last_entry_id="0-0",
                clear_on_success=True,
            )
            await self.stub.FunctionGetOutputs(request)


MAP_INVOCATION_CHUNK_SIZE = 49


async def _map_invocation(
    function_id: str,
    input_stream: AsyncIterable[Any],
    kwargs: Dict[str, Any],
    client: _Client,
    is_generator: bool,
    order_outputs: bool,
    return_exceptions: bool,
    count_update_callback: Optional[Callable[[int, int], None]],
):
    request = api_pb2.FunctionMapRequest(
        function_id=function_id,
        parent_input_id=current_input_id(),
        function_call_type=api_pb2.FUNCTION_CALL_TYPE_MAP,
        return_exceptions=return_exceptions,
    )
    response = await retry_transient_errors(client.stub.FunctionMap, request)

    function_call_id = response.function_call_id

    have_all_inputs = False
    num_inputs = 0
    num_outputs = 0
    pending_outputs: Dict[str, int] = {}  # Map input_id -> next expected gen_index value
    completed_outputs: Set[str] = set()  # Set of input_ids whose outputs are complete (expecting no more values)

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
            logger.debug(
                f"Pushing {len(items)} inputs to server. Num queued inputs awaiting push is {input_queue.qsize()}."
            )
            resp = await retry_transient_errors(
                client.stub.FunctionPutInputs,
                request,
                max_retries=None,
                max_delay=10,
                additional_status_codes=[Status.RESOURCE_EXHAUSTED],
            )
            for item in resp.inputs:
                pending_outputs.setdefault(item.input_id, 0)
            logger.debug(
                f"Successfully pushed {len(items)} inputs to server. Num queued inputs awaiting push is {input_queue.qsize()}."
            )

        have_all_inputs = True
        yield

    async def get_all_outputs():
        nonlocal num_inputs, num_outputs, have_all_inputs
        last_entry_id = "0-0"
        while not have_all_inputs or len(pending_outputs) > len(completed_outputs):
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=function_call_id,
                timeout=config["outputs_timeout"],
                last_entry_id=last_entry_id,
                clear_on_success=False,
            )
            response = await retry_transient_errors(
                client.stub.FunctionGetOutputs,
                request,
                max_retries=20,
                attempt_timeout=config["outputs_timeout"] + ATTEMPT_TIMEOUT_GRACE_PERIOD,
            )

            if len(response.outputs) == 0:
                continue

            last_entry_id = response.last_entry_id
            for item in response.outputs:
                pending_outputs.setdefault(item.input_id, 0)
                if item.input_id in completed_outputs or (
                    item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
                    and item.gen_index < pending_outputs[item.input_id]
                ):
                    # If this input is already completed, or if it's a generator output and we've already seen a later
                    # output, it means the output has already been processed and was received again due
                    # to a duplicate output enqueue on the server.
                    continue

                if is_generator:
                    # Mark this input completed if the generator completed successfully, or it crashed (exception, timeout, etc).
                    if item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                        completed_outputs.add(item.input_id)
                        num_outputs += 1
                    elif item.result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
                        completed_outputs.add(item.input_id)
                        num_outputs += 1
                        yield item
                    else:
                        assert pending_outputs[item.input_id] == item.gen_index
                        pending_outputs[item.input_id] += 1
                        yield item
                else:
                    completed_outputs.add(item.input_id)
                    num_outputs += 1
                    yield item

    async def get_all_outputs_and_clean_up():
        try:
            async for item in get_all_outputs():
                yield item
        finally:
            # "ack" that we have all outputs we are interested in and let backend clear results
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=function_call_id,
                timeout=0,
                last_entry_id="0-0",
                clear_on_success=True,
            )
            await retry_transient_errors(client.stub.FunctionGetOutputs, request)

    async def fetch_output(item: api_pb2.FunctionGetOutputsItem):
        try:
            output = await _process_result(item.result, item.data_format, client.stub, client)
        except Exception as e:
            if return_exceptions:
                output = e
            else:
                raise e
        return (item.idx, output)

    async def poll_outputs():
        outputs = stream.iterate(get_all_outputs_and_clean_up())
        outputs_fetched = outputs | pipe.map(fetch_output, ordered=True, task_limit=BLOB_MAX_PARALLELISM)

        # map to store out-of-order outputs received
        received_outputs = {}
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
                    received_outputs[idx] = output
                    while output_idx in received_outputs:
                        output = received_outputs.pop(output_idx)
                        yield _OutputValue(output)
                        output_idx += 1

        assert len(received_outputs) == 0

    response_gen = stream.merge(drain_input_generator(), pump_inputs(), poll_outputs())

    async with response_gen.stream() as streamer:
        async for response in streamer:
            if response is not None:
                yield response.value


# Wrapper type for api_pb2.FunctionStats
@dataclass
class FunctionStats:
    """Simple data structure storing stats for a running function."""

    backlog: int
    num_active_runners: int
    num_total_runners: int


class _Function(_Object, type_prefix="fu"):
    """Functions are the basic units of serverless execution on Modal.

    Generally, you will not construct a `Function` directly. Instead, use the
    `@stub.function()` decorator on the `Stub` object for your application.
    """

    # TODO: more type annotations
    _info: FunctionInfo
    _all_mounts: Collection[_Mount]
    _stub: "modal.stub._Stub"
    _obj: Any
    _web_url: Optional[str]
    _is_remote_cls_method: bool = False  # TODO(erikbern): deprecated
    _function_name: Optional[str]
    _is_method: bool

    @staticmethod
    def from_args(
        info: FunctionInfo,
        stub,
        image=None,
        secret: Optional[_Secret] = None,
        secrets: Collection[_Secret] = (),
        schedule: Optional[Schedule] = None,
        is_generator=False,
        gpu: GPU_T = None,
        # TODO: maybe break this out into a separate decorator for notebooks.
        mounts: Collection[_Mount] = (),
        network_file_systems: Dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        allow_cross_region_volumes: bool = False,
        volumes: Dict[Union[str, os.PathLike], _Volume] = {},
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
        memory: Optional[int] = None,
        proxy: Optional[_Proxy] = None,
        retries: Optional[Union[int, Retries]] = None,
        timeout: Optional[int] = None,
        concurrency_limit: Optional[int] = None,
        allow_concurrent_inputs: Optional[int] = None,
        container_idle_timeout: Optional[int] = None,
        cpu: Optional[float] = None,
        keep_warm: Optional[int] = None,
        interactive: bool = False,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        is_builder_function: bool = False,
        cls: Optional[type] = None,
        is_auto_snapshot: bool = False,
        checkpointing_enabled: bool = False,
        allow_background_volume_commits: bool = False,
    ) -> None:
        """mdmd:hidden"""
        tag = info.get_tag()

        raw_f = info.raw_f
        assert callable(raw_f)
        if schedule is not None:
            if not info.is_nullary():
                raise InvalidError(
                    f"Function {raw_f} has a schedule, so it needs to support calling it with no arguments"
                )

        all_mounts = [
            _get_client_mount(),  # client
            *mounts,  # explicit mounts
        ]
        # TODO (elias): Clean up mount logic, this is quite messy:
        if stub:
            all_mounts.extend(stub._get_deduplicated_function_mounts(info.get_mounts()))  # implicit mounts
        else:
            all_mounts.extend(info.get_mounts().values())  # this would typically only happen for builder functions

        if secret:
            secrets = [secret, *secrets]

        if isinstance(retries, int):
            retry_policy = Retries(
                max_retries=retries,
                initial_delay=1.0,
                backoff_coefficient=1.0,
            )._to_proto()
        elif isinstance(retries, Retries):
            retry_policy = retries._to_proto()
        elif retries is None:
            retry_policy = None
        else:
            raise InvalidError(
                f"Function {raw_f} retries must be an integer or instance of modal.Retries. Found: {type(retries)}"
            )

        gpu_config = parse_gpu_config(gpu)

        if proxy:
            # HACK: remove this once we stop using ssh tunnels for this.
            if image:
                image = image.apt_install("autossh")

        if not is_auto_snapshot and (hasattr(info.cls, "__build__") or hasattr(info.cls, "__abuild__")):
            if hasattr(info.cls, "__build__"):
                snapshot_info = FunctionInfo(info.cls.__build__, cls=info.cls)
            else:
                snapshot_info = FunctionInfo(info.cls.__abuild__, cls=info.cls)

            snapshot_function = _Function.from_args(
                snapshot_info,
                stub=None,
                image=image,
                secret=secret,
                secrets=secrets,
                gpu=gpu,
                mounts=mounts,
                network_file_systems=network_file_systems,
                volumes=volumes,
                memory=memory,
                timeout=timeout,
                cpu=cpu,
                is_builder_function=True,
                is_auto_snapshot=True,
            )
            image = image.extend(build_function=snapshot_function, force_build=image.force_build)

        if interactive and concurrency_limit and concurrency_limit > 1:
            warnings.warn(
                "Interactive functions require `concurrency_limit=1`. The concurrency limit will be overridden."
            )
            concurrency_limit = 1

        if keep_warm is not None:
            assert isinstance(keep_warm, int)

        if not cloud and not is_builder_function:
            cloud = config.get("default_cloud")
        if cloud:
            cloud_provider = parse_cloud_provider(cloud)
        else:
            cloud_provider = None

        if is_generator and webhook_config:
            if webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
                raise InvalidError(
                    """Webhooks cannot be generators. If you want a streaming response, see https://modal.com/docs/guide/streaming-endpoints
                    """
                )
            else:
                raise InvalidError("Webhooks cannot be generators")

        # Validate volumes
        if not isinstance(volumes, dict):
            raise InvalidError("volumes must be a dict[str, Volume] where the keys are paths")
        validated_volumes = validate_mount_points("Volume", volumes)
        # We don't support mounting a volume in more than one location
        volume_to_paths: Dict[_Volume, List[str]] = {}
        for path, volume in validated_volumes:
            volume_to_paths.setdefault(volume, []).append(path)
        for paths in volume_to_paths.values():
            if len(paths) > 1:
                conflicting = ", ".join(paths)
                raise InvalidError(
                    f"The same Volume cannot be mounted in multiple locations for the same function: {conflicting}"
                )

        # Validate NFS
        if not isinstance(network_file_systems, dict):
            raise InvalidError("network_file_systems must be a dict[str, NetworkFileSystem] where the keys are paths")
        validated_network_file_systems = validate_mount_points("Network file system", network_file_systems)

        # Validate image
        if image is not None and not isinstance(image, _Image):
            raise InvalidError(f"Expected modal.Image object. Got {type(image)}.")

        def _deps(only_explicit_mounts=False) -> List[_Object]:
            deps: List[_Object] = list(secrets)
            if only_explicit_mounts:
                # TODO: this is a bit hacky, but all_mounts may differ in the container vs locally
                # We don't want the function dependencies to change, so we have this way to force it to
                # only include its declared dependencies
                deps += list(mounts)
            else:
                deps += list(all_mounts)
            if proxy:
                deps.append(proxy)
            if image:
                deps.append(image)
            for _, nfs in validated_network_file_systems:
                deps.append(nfs)
            for _, vol in validated_volumes:
                deps.append(vol)
            return deps

        async def _preload(provider: _Function, resolver: Resolver, existing_object_id: Optional[str]):
            if is_generator:
                function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
            else:
                function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

            req = api_pb2.FunctionPrecreateRequest(
                app_id=resolver.app_id,
                function_name=info.function_name,
                function_type=function_type,
                webhook_config=webhook_config,
                existing_function_id=existing_object_id,
            )
            response = await retry_transient_errors(resolver.client.stub.FunctionPrecreate, req)
            # Update the precreated function handle (todo: hack until we merge providers/handles)
            provider._hydrate(response.function_id, resolver.client, response.handle_metadata)

        async def _load(provider: _Object, resolver: Resolver, existing_object_id: Optional[str]):
            status_row = resolver.add_status_row()
            status_row.message(f"Creating {tag}...")

            if is_generator:
                function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
            else:
                function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

            if cpu is not None and cpu < 0.25:
                raise InvalidError(f"Invalid fractional CPU value {cpu}. Cannot have less than 0.25 CPU resources.")
            milli_cpu = int(1000 * cpu) if cpu is not None else None

            timeout_secs = timeout
            if resolver.shell and not is_builder_function:
                timeout_secs = 86400
                pty_info = _pty.get_pty_info(shell=True)
            elif interactive:
                assert not is_builder_function, "builder functions do not support interactive usage"
                pty_info = _pty.get_pty_info(shell=False)
            else:
                pty_info = None

            if info.is_serialized():
                # Use cloudpickle. Used when working w/ Jupyter notebooks.
                # serialize at _load time, not function decoration time
                # otherwise we can't capture a surrounding class for lifetime methods etc.
                function_serialized = info.serialized_function()
                class_serialized = serialize(cls) if cls is not None else None

                # Ensure that large data in global variables does not blow up the gRPC payload,
                # which has maximum size 100 MiB. We set the limit lower for performance reasons.
                if len(function_serialized) > 16 << 20:  # 16 MiB
                    raise InvalidError(
                        f"Function {info.raw_f} has size {len(function_serialized)} bytes when packaged. "
                        "This is larger than the maximum limit of 16 MiB. "
                        "Try reducing the size of the closure by using parameters or mounts, not large global variables."
                    )
                elif len(function_serialized) > 256 << 10:  # 256 KiB
                    warnings.warn(
                        f"Function {info.raw_f} has size {len(function_serialized)} bytes when packaged. "
                        "This is larger than the recommended limit of 256 KiB. "
                        "Try reducing the size of the closure by using parameters or mounts, not large global variables."
                    )
            else:
                function_serialized = None
                class_serialized = None

            stub_name = ""
            if stub and stub.name:
                stub_name = stub.name

            # Relies on dicts being ordered (true as of Python 3.6).
            volume_mounts = [
                api_pb2.VolumeMount(
                    mount_path=path,
                    volume_id=volume.object_id,
                    allow_background_commits=allow_background_volume_commits,
                )
                for path, volume in validated_volumes
            ]

            # Create function remotely
            function_definition = api_pb2.Function(
                module_name=info.module_name,
                function_name=info.function_name,
                mount_ids=[mount.object_id for mount in all_mounts],
                secret_ids=[secret.object_id for secret in secrets],
                image_id=(image.object_id if image else None),
                definition_type=info.definition_type,
                function_serialized=function_serialized,
                class_serialized=class_serialized,
                function_type=function_type,
                resources=api_pb2.Resources(milli_cpu=milli_cpu, gpu_config=gpu_config, memory_mb=memory),
                webhook_config=webhook_config,
                shared_volume_mounts=network_file_system_mount_protos(
                    validated_network_file_systems, allow_cross_region_volumes
                ),
                volume_mounts=volume_mounts,
                proxy_id=(proxy.object_id if proxy else None),
                retry_policy=retry_policy,
                timeout_secs=timeout_secs,
                task_idle_timeout_secs=container_idle_timeout,
                concurrency_limit=concurrency_limit,
                pty_info=pty_info,
                cloud_provider=cloud_provider,
                warm_pool_size=keep_warm,
                runtime=config.get("function_runtime"),
                runtime_debug=config.get("function_runtime_debug"),
                stub_name=stub_name,
                is_builder_function=is_builder_function,
                allow_concurrent_inputs=allow_concurrent_inputs,
                worker_id=config.get("worker_id"),
                is_auto_snapshot=is_auto_snapshot,
                is_method=bool(cls),
                checkpointing_enabled=checkpointing_enabled,
                is_checkpointing_function=False,
                object_dependencies=[
                    api_pb2.ObjectDependency(object_id=dep.object_id) for dep in _deps(only_explicit_mounts=True)
                ],
            )
            request = api_pb2.FunctionCreateRequest(
                app_id=resolver.app_id,
                function=function_definition,
                schedule=schedule.proto_message if schedule is not None else None,
                existing_function_id=existing_object_id,
            )
            try:
                response: api_pb2.FunctionCreateResponse = await retry_transient_errors(
                    resolver.client.stub.FunctionCreate, request
                )
            except GRPCError as exc:
                if exc.status == Status.INVALID_ARGUMENT:
                    raise InvalidError(exc.message)
                if exc.status == Status.FAILED_PRECONDITION:
                    raise InvalidError(exc.message)
                if "Received :status = '413'" in exc.message:
                    raise InvalidError(f"Function {raw_f} is too large to deploy.")
                raise

            if response.function.web_url:
                # Ensure terms used here match terms used in modal.com/docs/guide/webhook-urls doc.
                if response.function.web_url_info.truncated:
                    suffix = " [grey70](label truncated)[/grey70]"
                elif response.function.web_url_info.has_unique_hash:
                    suffix = " [grey70](label includes conflict-avoidance hash)[/grey70]"
                elif response.function.web_url_info.label_stolen:
                    suffix = " [grey70](label stolen)[/grey70]"
                else:
                    suffix = ""
                # TODO: this is only printed when we're showing progress. Maybe move this somewhere else.
                status_row.finish(f"Created {tag} => [magenta underline]{response.web_url}[/magenta underline]{suffix}")

                # Print custom domain in terminal
                for custom_domain in response.function.custom_domain_info:
                    custom_domain_status_row = resolver.add_status_row()
                    custom_domain_status_row.finish(
                        f"Custom domain for {tag} => [magenta underline]{custom_domain.url}[/magenta underline]{suffix}"
                    )

            else:
                status_row.finish(f"Created {tag}.")

            provider._hydrate(response.function_id, resolver.client, response.handle_metadata)

        rep = f"Function({tag})"
        obj = _Function._from_loader(_load, rep, preload=_preload, deps=_deps)

        obj._raw_f = raw_f
        obj._info = info
        obj._tag = tag
        obj._all_mounts = all_mounts  # needed for modal.serve file watching
        obj._stub = stub  # Needed for CLI right now
        obj._obj = None
        obj._is_generator = is_generator
        obj._is_method = bool(cls)

        # Used to check whether we should rebuild an image using run_function
        # Plaintext source and arg definition for the function, so it's part of the image
        # hash. We can't use the cloudpickle hash because it's not very stable.
        obj._build_args = dict(  # See get_build_def
            secrets=repr(secrets),
            gpu_config=repr(gpu_config),
            mounts=repr(mounts),
            network_file_systems=repr(network_file_systems),
        )

        return obj

    def from_parametrized(self, obj, args: Iterable[Any], kwargs: Dict[str, Any]) -> "_Function":
        async def _load(provider: _Function, resolver: Resolver, existing_object_id: Optional[str]):
            if not self.is_hydrated:
                raise ExecutionError(
                    "Base function in class has not been hydrated. This might happen if an object is"
                    " defined on a different stub, or if it's on the same stub but it didn't get"
                    " created because it wasn't defined in global scope."
                )
            serialized_params = pickle.dumps((args, kwargs))  # TODO(erikbern): use modal._serialization?
            req = api_pb2.FunctionBindParamsRequest(
                function_id=self._object_id,
                serialized_params=serialized_params,
            )
            response = await retry_transient_errors(self._client.stub.FunctionBindParams, req)
            provider._hydrate(response.bound_function_id, self._client, response.handle_metadata)

        provider = _Function._from_loader(_load, "Function(parametrized)", hydrate_lazily=True)
        if len(args) + len(kwargs) == 0 and self.is_hydrated:
            # Edge case that lets us hydrate all objects right away
            provider._hydrate_from_other(self)
        provider._is_remote_cls_method = True  # TODO(erikbern): deprecated
        provider._info = self._info
        provider._obj = obj
        provider._is_generator = self._is_generator
        provider._is_method = True

        return provider

    @property
    def tag(self):
        """mdmd:hidden"""
        return self._tag

    @property
    def stub(self) -> "modal.stub._Stub":
        return self._stub

    @property
    def info(self) -> FunctionInfo:
        return self._info

    def get_build_def(self) -> str:
        """mdmd:hidden"""
        return f"{inspect.getsource(self._raw_f)}\n{repr(self._build_args)}"

    # Live handle methods

    def _initialize_from_empty(self):
        # Overridden concrete implementation of base class method
        self._progress = None
        self._is_generator = None
        self._web_url = None
        self._output_mgr: Optional[OutputManager] = None
        self._mute_cancellation = (
            False  # set when a user terminates the app intentionally, to prevent useless traceback spam
        )
        self._function_name = None
        self._info = None

    def _hydrate_metadata(self, metadata: Message):
        # Overridden concrete implementation of base class method
        assert isinstance(metadata, (api_pb2.Function, api_pb2.FunctionHandleMetadata))
        self._is_generator = metadata.function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR
        self._web_url = metadata.web_url
        self._function_name = metadata.function_name
        self._is_method = metadata.is_method

    def _get_metadata(self):
        # Overridden concrete implementation of base class method
        return api_pb2.FunctionHandleMetadata(
            function_name=self._function_name,
            function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR
            if self._is_generator
            else api_pb2.Function.FUNCTION_TYPE_FUNCTION,
            web_url=self._web_url,
        )

    def _set_mute_cancellation(self, value: bool = True):
        self._mute_cancellation = value

    def _set_output_mgr(self, output_mgr: OutputManager):
        self._output_mgr = output_mgr

    @property
    def web_url(self) -> str:
        """URL of a Function running as a web endpoint."""
        return self._web_url

    @property
    def is_generator(self) -> bool:
        return self._is_generator

    async def _map(self, input_stream: AsyncIterable[Any], order_outputs: bool, return_exceptions: bool, kwargs={}):
        if self._web_url:
            raise InvalidError(
                "A web endpoint function cannot be directly invoked for parallel remote execution. "
                f"Invoke this function via its web url '{self._web_url}' or call it locally: {self._function_name}()."
            )

        if order_outputs and self._is_generator:
            raise ValueError("Can't return ordered results for a generator")

        count_update_callback = (
            self._output_mgr.function_progress_callback(self._function_name, total=None) if self._output_mgr else None
        )

        async for item in _map_invocation(
            self.object_id,
            input_stream,
            kwargs,
            self._client,
            self._is_generator,
            order_outputs,
            return_exceptions,
            count_update_callback,
        ):
            yield item

    async def _call_function(self, args, kwargs):
        invocation = await _Invocation.create(self.object_id, args, kwargs, self._client)
        try:
            return await invocation.run_function()
        except asyncio.CancelledError:
            # this can happen if the user terminates a program, triggering a cancellation cascade
            if not self._mute_cancellation:
                raise

    async def _call_function_nowait(self, args, kwargs):
        return await _Invocation.create(self.object_id, args, kwargs, self._client)

    @warn_if_generator_is_not_consumed
    @live_method_gen
    async def _call_generator(self, args, kwargs):
        invocation = await _Invocation.create(self.object_id, args, kwargs, self._client)
        async for res in invocation.run_generator():
            yield res

    async def _call_generator_nowait(self, args, kwargs):
        return await _Invocation.create(self.object_id, args, kwargs, self._client)

    @warn_if_generator_is_not_consumed
    @live_method_gen
    async def map(
        self,
        *input_iterators,  # one input iterator per argument in the mapped-over function/generator
        kwargs={},  # any extra keyword arguments for the function
        order_outputs=None,  # defaults to True for regular functions, False for generators
        return_exceptions=False,  # whether to propogate exceptions (False) or aggregate them in the results list (True)
    ) -> AsyncGenerator[Any, None]:
        """Parallel map over a set of inputs.

        Takes one iterator argument per argument in the function being mapped over.

        Example:
        ```python
        @stub.function()
        def my_func(a):
            return a ** 2

        @stub.local_entrypoint()
        def main():
            assert list(my_func.map([1, 2, 3, 4])) == [1, 4, 9, 16]
        ```

        If applied to a `stub.function`, `map()` returns one result per input and the output order
        is guaranteed to be the same as the input order. Set `order_outputs=False` to return results
        in the order that they are completed instead.

        By yielding zero or more than once, mapping over generators can also be used
        as a "flat map".

        ```python
        @stub.function()
        def flat(i: int):
            choices = [[], ["once"], ["two", "times"], ["three", "times", "a lady"]]
            for item in choices[i % len(choices)]:
                yield item

        @stub.local_entrypoint()
        def main():
            for item in flat.map(range(10)):
                print(item)
        ```

        `return_exceptions` can be used to treat exceptions as successful results:
        ```python
        @stub.function()
        def my_func(a):
            if a == 2:
                raise Exception("ohno")
            return a ** 2

        @stub.local_entrypoint()
        def main():
            # [0, 1, UserCodeException(Exception('ohno'))]
            print(list(my_func.map(range(3), return_exceptions=True)))
        ```
        """
        if order_outputs is None:
            order_outputs = not self._is_generator

        input_stream = stream.zip(*(stream.iterate(it) for it in input_iterators))
        async for item in self._map(input_stream, order_outputs, return_exceptions, kwargs):
            yield item

    async def for_each(self, *input_iterators, kwargs={}, ignore_exceptions=False):
        """Execute function for all outputs, ignoring outputs

        Convenient alias for `.map()` in cases where the function just needs to be called.
        as the caller doesn't have to consume the generator to process the inputs.
        """
        # TODO(erikbern): it would be better if this is more like a map_spawn that immediately exits
        # rather than iterating over the result
        async for _ in self.map(
            *input_iterators, kwargs=kwargs, order_outputs=False, return_exceptions=ignore_exceptions
        ):
            pass

    @warn_if_generator_is_not_consumed
    @live_method_gen
    async def starmap(
        self, input_iterator, kwargs={}, order_outputs=None, return_exceptions=False
    ) -> AsyncGenerator[Any, None]:
        """Like `map` but spreads arguments over multiple function arguments

        Assumes every input is a sequence (e.g. a tuple).

        Example:
        ```python
        @stub.function()
        def my_func(a, b):
            return a + b

        @stub.local_entrypoint()
        def main():
            assert list(my_func.starmap([(1, 2), (3, 4)])) == [3, 7]
        ```
        """
        if order_outputs is None:
            order_outputs = not self._is_generator

        input_stream = stream.iterate(input_iterator)
        async for item in self._map(input_stream, order_outputs, return_exceptions, kwargs):
            yield item

    @live_method
    async def remote(self, *args, **kwargs) -> Awaitable[Any]:
        """
        Calls the function remotely, executing it with the given arguments and returning the execution's result.
        """
        # TODO: Generics/TypeVars
        if self._web_url:
            raise InvalidError(
                "A web endpoint function cannot be invoked for remote execution with `.remote`. "
                f"Invoke this function via its web url '{self._web_url}' or call it locally: {self._function_name}()."
            )
        if self._is_generator:
            raise InvalidError(
                "A generator function cannot be called with `.remote(...)`. Use `.remote_gen(...)` instead."
            )

        return await self._call_function(args, kwargs)

    @live_method_gen
    async def remote_gen(self, *args, **kwargs) -> AsyncGenerator[Any, None]:
        """
        Calls the generator remotely, executing it with the given arguments and returning the execution's result.
        """
        # TODO: Generics/TypeVars
        if self._web_url:
            raise InvalidError(
                "A web endpoint function cannot be invoked for remote execution with `.remote`. "
                f"Invoke this function via its web url '{self._web_url}' or call it locally: {self._function_name}()."
            )

        if not self._is_generator:
            raise InvalidError(
                "A non-generator function cannot be called with `.remote_gen(...)`. Use `.remote(...)` instead."
            )
        async for item in self._call_generator(args, kwargs):  # type: ignore
            yield item

    def call(self, *args, **kwargs) -> Awaitable[Any]:
        """Deprecated. Use `f.remote` or `f.remote_gen` instead."""
        # TODO: Generics/TypeVars
        if self._is_generator:
            deprecation_error(
                date(2023, 8, 16), "`f.call(...)` is deprecated. It has been renamed to `f.remote_gen(...)`"
            )
        else:
            deprecation_error(date(2023, 8, 16), "`f.call(...)` is deprecated. It has been renamed to `f.remote(...)`")

    @live_method
    async def shell(self, *args, **kwargs) -> None:
        if self._is_generator:
            async for item in self._call_generator(args, kwargs):
                pass
        else:
            await self._call_function(args, kwargs)

    def _get_is_remote_cls_method(self):
        return self._is_remote_cls_method

    def _get_info(self):
        return self._info

    def _get_obj(self):
        if not self._is_method:
            return None
        elif not self._obj:
            raise ExecutionError("Method has no local object")
        else:
            return self._obj

    @synchronizer.nowrap
    def local(self, *args, **kwargs) -> Any:
        """
        Calls the function locally, executing it with the given arguments and returning the execution's result.
        This method allows a caller to execute the standard Python function wrapped by Modal.
        """
        # TODO(erikbern): it would be nice to remove the nowrap thing, but right now that would cause
        # "user code" to run on the synchronicity thread, which seems bad
        info = self._get_info()
        if not info:
            msg = (
                "The definition for this function is missing so it is not possible to invoke it locally. "
                "If this function was retrieved via `Function.lookup` you need to use `.remote()`."
            )
            raise ExecutionError(msg)

        obj = self._get_obj()

        if not obj:
            fun = info.raw_f
            return fun(*args, **kwargs)
        else:
            # This is a method on a class, so bind the self to the function
            local_obj = obj.get_local_obj()
            fun = info.raw_f.__get__(local_obj)

            if is_async(info.raw_f):
                # We want to run __aenter__ and fun in the same coroutine
                async def coro():
                    await obj.aenter()
                    return await fun(*args, **kwargs)

                return coro()
            else:
                obj.enter()
                return fun(*args, **kwargs)

    @synchronizer.nowrap
    def __call__(self, *args, **kwargs) -> Any:  # TODO: Generics/TypeVars
        if self._get_is_remote_cls_method():
            deprecation_error(
                date(2023, 9, 1),
                "Calling remote class methods like `obj.f(...)` is deprecated. Use `obj.f.remote(...)` for remote calls"
                " and `obj.f.local(...)` for local calls",
            )
        else:
            deprecation_error(
                date(2023, 8, 16),
                "Calling Modal functions like `f(...)` is deprecated. Use `f.local(...)` if you want to call the"
                " function in the same Python process. Use `f.remote(...)` if you want to call the function in"
                " a Modal container in the cloud",
            )

    @live_method
    async def spawn(self, *args, **kwargs) -> Optional["_FunctionCall"]:
        """Calls the function with the given arguments, without waiting for the results.

        Returns a `modal.functions.FunctionCall` object, that can later be polled or waited for using `.get(timeout=...)`.
        Conceptually similar to `multiprocessing.pool.apply_async`, or a Future/Promise in other contexts.

        *Note:* `.spawn()` on a modal generator function does call and execute the generator, but does not currently
        return a function handle for polling the result.
        """
        if self._is_generator:
            await self._call_generator_nowait(args, kwargs)
            return None

        invocation = await self._call_function_nowait(args, kwargs)
        return _FunctionCall._new_hydrated(invocation.function_call_id, invocation.client, None)

    def get_raw_f(self) -> Callable[..., Any]:
        """Return the inner Python object wrapped by this Modal Function."""
        if not self._info:
            raise AttributeError("_info has not been set on this FunctionHandle and not available in this context")

        return self._info.raw_f

    @live_method
    async def get_current_stats(self) -> FunctionStats:
        """Return a `FunctionStats` object describing the current function's queue and runner counts."""

        resp = await self._client.stub.FunctionGetCurrentStats(
            api_pb2.FunctionGetCurrentStatsRequest(function_id=self.object_id)
        )
        return FunctionStats(
            backlog=resp.backlog, num_active_runners=resp.num_active_tasks, num_total_runners=resp.num_total_tasks
        )


Function = synchronize_api(_Function)


class _FunctionCall(_Object, type_prefix="fc"):
    """A reference to an executed function call.

    Constructed using `.spawn(...)` on a Modal function with the same
    arguments that a function normally takes. Acts as a reference to
    an ongoing function call that can be passed around and used to
    poll or fetch function results at some later time.

    Conceptually similar to a Future/Promise/AsyncResult in other contexts and languages.
    """

    def _invocation(self):
        assert self._client
        return _Invocation(self._client.stub, self.object_id, self._client)

    async def get(self, timeout: Optional[float] = None):
        """Get the result of the function call.

        This function waits indefinitely by default. It takes an optional
        `timeout` argument that specifies the maximum number of seconds to wait,
        which can be set to `0` to poll for an output immediately.

        The returned coroutine is not cancellation-safe.
        """
        return await self._invocation().poll_function(timeout=timeout)

    async def get_call_graph(self) -> List[InputInfo]:
        """Returns a structure representing the call graph from a given root
        call ID, along with the status of execution for each node.

        See [`modal.call_graph`](/docs/reference/modal.call_graph) reference page
        for documentation on the structure of the returned `InputInfo` items.
        """
        assert self._client and self._client.stub
        request = api_pb2.FunctionGetCallGraphRequest(function_call_id=self.object_id)
        response = await retry_transient_errors(self._client.stub.FunctionGetCallGraph, request)
        return _reconstruct_call_graph(response)

    async def cancel(self):
        request = api_pb2.FunctionCallCancelRequest(function_call_id=self.object_id)
        assert self._client and self._client.stub
        await retry_transient_errors(self._client.stub.FunctionCallCancel, request)


FunctionCall = synchronize_api(_FunctionCall)


async def _gather(*function_calls: _FunctionCall):
    """Wait until all Modal function calls have results before returning

    Accepts a variable number of FunctionCall objects as returned by `Function.spawn()`.

    Returns a list of results from each function call, or raises an exception
    of the first failing function call.

    E.g.

    ```python notest
    function_call_1 = slow_func_1.spawn()
    function_call_2 = slow_func_2.spawn()

    result_1, result_2 = gather(function_call_1, function_call_2)
    ```
    """
    try:
        return await asyncio.gather(*[fc.get() for fc in function_calls])
    except Exception as exc:
        # TODO: kill all running function calls
        raise exc


gather = synchronize_api(_gather)


_current_input_id: Optional[str] = None
_current_function_call_id: Optional[str] = None


def current_input_id() -> str:
    """Returns the input ID for the current input.

    Can only be called from Modal function (i.e. in a container context).

    ```python
    from modal import current_input_id

    @stub.function()
    def process_stuff():
        print(f"Starting to process {current_input_id()}")
    ```
    """
    global _current_input_id
    return _current_input_id


def current_function_call_id() -> str:
    """Returns the function call ID for the current input.

    Can only be called from Modal function (i.e. in a container context).

    ```python
    from modal import current_function_call_id

    @stub.function()
    def process_stuff():
        print(f"Starting to process input from {current_function_call_id()}")
    ```
    """
    global _current_function_call_id
    return _current_function_call_id


def _set_current_input_id(input_id: Optional[str]):
    global _current_input_id
    _current_input_id = input_id


def _set_current_function_call_id(function_call_id: Optional[str]):
    global _current_function_call_id
    _current_function_call_id = function_call_id


class _PartialFunction:
    """Intermediate function, produced by @method or @web_endpoint"""

    def __init__(
        self,
        raw_f: Callable[..., Any],
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
        is_generator: Optional[bool] = None,
        keep_warm: Optional[int] = None,
    ):
        self.raw_f = raw_f
        self.webhook_config = webhook_config
        self.is_generator = is_generator
        self.keep_warm = keep_warm
        self.wrapped = False  # Make sure that this was converted into a FunctionHandle

    def __get__(self, obj, objtype=None) -> _Function:
        # This only happens inside user methods when they refer to other methods
        k = self.raw_f.__name__
        if obj:  # Cls().fun
            function = getattr(obj, "_modal_functions")[k]
        else:  # Cls.fun
            function = getattr(objtype, "_modal_functions")[k]
        return function

    def __del__(self):
        if self.wrapped is False:
            logger.warning(
                f"Method or web function {self.raw_f} was never turned into a function."
                " Did you forget a @stub.function or @stub.cls decorator?"
            )


PartialFunction = synchronize_api(_PartialFunction)


def _method(
    _warn_parentheses_missing=None,
    *,
    # Set this to True if it's a non-generator function returning
    # a [sync/async] generator object
    is_generator: Optional[bool] = None,
    keep_warm: Optional[int] = None,  # An optional number of containers to always keep warm.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator for methods that should be transformed into a Modal Function registered against this class's stub.

    **Usage:**

    ```python
    @stub.cls(cpu=8)
    class MyCls:

        @modal.method()
        def f(self):
            ...
    ```
    """
    if _warn_parentheses_missing:
        raise InvalidError("Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@method()`.")

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if isinstance(raw_f, _PartialFunction) and raw_f.webhook_config:
            raw_f.wrapped = True  # suppress later warning
            raise InvalidError(
                "Web endpoints on classes should not be wrapped by `@method`. Suggestion: remove the `@method` decorator."
            )
        return _PartialFunction(raw_f, is_generator=is_generator, keep_warm=keep_warm)

    return wrapper


def _parse_custom_domains(custom_domains: Optional[Iterable[str]] = None) -> List[api_pb2.CustomDomainConfig]:
    _custom_domains: List[api_pb2.CustomDomainConfig] = []
    if custom_domains is not None:
        for custom_domain in custom_domains:
            _custom_domains.append(api_pb2.CustomDomainConfig(name=custom_domain))

    return _custom_domains


@typechecked
def _web_endpoint(
    _warn_parentheses_missing=None,
    *,
    method: str = "GET",  # REST method for the created endpoint.
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    custom_domains: Optional[
        Iterable[str]
    ] = None,  # Create an endpoint using a custom domain fully-qualified domain name.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Register a basic web endpoint with this application.

    This is the simple way to create a web endpoint on Modal. The function
    behaves as a [FastAPI](https://fastapi.tiangolo.com/) handler and should
    return a response object to the caller.

    Endpoints created with `@stub.web_endpoint` are meant to be simple, single
    request handlers and automatically have
    [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) enabled.
    For more flexibility, use `@stub.asgi_app`.

    To learn how to use Modal with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).
    """
    if isinstance(_warn_parentheses_missing, str):
        # Probably passing the method string as a positional argument.
        raise InvalidError('Positional arguments are not allowed. Suggestion: `@web_endpoint(method="GET")`.')
    elif _warn_parentheses_missing:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@web_endpoint()`."
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if isinstance(raw_f, _Function):
            raw_f = raw_f.get_raw_f()
            raise InvalidError(
                f"Applying decorators for {raw_f} in the wrong order!\nUsage:\n\n"
                "@stub.function()\n@stub.web_endpoint()\ndef my_webhook():\n    ..."
            )
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        # self._loose_webhook_configs.add(raw_f)

        return _PartialFunction(
            raw_f,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_FUNCTION,
                method=method,
                requested_suffix=label,
                async_mode=_response_mode,
                custom_domains=_parse_custom_domains(custom_domains),
            ),
        )

    return wrapper


@typechecked
def _asgi_app(
    _warn_parentheses_missing=None,
    *,
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    custom_domains: Optional[
        Iterable[str]
    ] = None,  # Create an endpoint using a custom domain fully-qualified domain name.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator for registering an ASGI app with a Modal function.

    Asynchronous Server Gateway Interface (ASGI) is a standard for Python
    synchronous and asynchronous apps, supported by all popular Python web
    libraries. This is an advanced decorator that gives full flexibility in
    defining one or more web endpoints on Modal.

    **Usage:**

    ```python
    from typing import Callable

    @stub.function()
    @modal.asgi_app()
    def create_asgi() -> Callable:
        ...
    ```

    To learn how to use Modal with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).
    """
    if isinstance(_warn_parentheses_missing, str):
        raise InvalidError('Positional arguments are not allowed. Suggestion: `@asgi_app(label="foo")`.')
    elif _warn_parentheses_missing:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@asgi_app()`."
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        return _PartialFunction(
            raw_f,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_ASGI_APP,
                requested_suffix=label,
                async_mode=_response_mode,
                custom_domains=_parse_custom_domains(custom_domains),
            ),
        )

    return wrapper


@typechecked
def _wsgi_app(
    _warn_parentheses_missing=None,
    *,
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    custom_domains: Optional[
        Iterable[str]
    ] = None,  # Create an endpoint using a custom domain fully-qualified domain name.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator for registering a WSGI app with a Modal function.

    Web Server Gateway Interface (WSGI) is a standard for synchronous Python web apps.
    It has been [succeeded by the ASGI interface](https://asgi.readthedocs.io/en/latest/introduction.html#wsgi-compatibility) which is compatible with ASGI and supports
    additional functionality such as web sockets. Modal supports ASGI via [`asgi_app`](/docs/reference/modal.asgi_app).

    **Usage:**

    ```python
    from typing import Callable

    @stub.function()
    @modal.wsgi_app()
    def create_wsgi() -> Callable:
        ...
    ```

    To learn how to use this decorator with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).
    """
    if isinstance(_warn_parentheses_missing, str):
        raise InvalidError('Positional arguments are not allowed. Suggestion: `@wsgi_app(label="foo")`.')
    elif _warn_parentheses_missing:
        raise InvalidError(
            "Positional arguments are not allowed. Did you forget parentheses? Suggestion: `@wsgi_app()`."
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        return _PartialFunction(
            raw_f,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_WSGI_APP,
                requested_suffix=label,
                async_mode=_response_mode,
                custom_domains=_parse_custom_domains(custom_domains),
            ),
        )

    return wrapper


method = synchronize_api(_method)
web_endpoint = synchronize_api(_web_endpoint)
asgi_app = synchronize_api(_asgi_app)
wsgi_app = synchronize_api(_wsgi_app)
