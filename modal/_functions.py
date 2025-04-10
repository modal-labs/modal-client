# Copyright Modal Labs 2023
import asyncio
import dataclasses
import inspect
import textwrap
import time
import typing
import warnings
from collections.abc import AsyncGenerator, Collection, Sequence, Sized
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import typing_extensions
from google.protobuf.message import Message
from grpclib import GRPCError, Status
from synchronicity.combined_types import MethodWithAio
from synchronicity.exceptions import UserCodeException

from modal_proto import api_pb2
from modal_proto.modal_api_grpc import ModalClientModal

from ._object import _get_environment_name, _Object, live_method, live_method_gen
from ._pty import get_pty_info
from ._resolver import Resolver
from ._resources import convert_fn_config_to_resources_config
from ._runtime.execution_context import current_input_id, is_local
from ._serialization import (
    apply_defaults,
    get_callable_schema,
    serialize,
    serialize_proto_params,
    validate_parameter_values,
)
from ._traceback import print_server_warnings
from ._utils.async_utils import (
    TaskContext,
    aclosing,
    async_merge,
    callable_to_agen,
    synchronizer,
    warn_if_generator_is_not_consumed,
)
from ._utils.deprecation import deprecation_error, deprecation_warning, renamed_parameter
from ._utils.function_utils import (
    ATTEMPT_TIMEOUT_GRACE_PERIOD,
    OUTPUTS_TIMEOUT,
    FunctionCreationStatus,
    FunctionInfo,
    IncludeSourceMode,
    _create_input,
    _process_result,
    _stream_function_call_data,
    get_function_type,
    get_include_source_mode,
    is_async,
)
from ._utils.grpc_utils import retry_transient_errors
from ._utils.mount_utils import validate_network_file_systems, validate_volumes
from .call_graph import InputInfo, _reconstruct_call_graph
from .client import _Client
from .cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from .config import config
from .exception import (
    ExecutionError,
    FunctionTimeoutError,
    InternalFailure,
    InvalidError,
    NotFoundError,
    OutputExpiredError,
)
from .gpu import GPU_T, parse_gpu_config
from .image import _Image
from .mount import _get_client_mount, _Mount, get_sys_modules_mounts
from .network_file_system import _NetworkFileSystem, network_file_system_mount_protos
from .output import _get_output_manager
from .parallel_map import (
    _for_each_async,
    _for_each_sync,
    _map_async,
    _map_invocation,
    _map_sync,
    _starmap_async,
    _starmap_sync,
    _SynchronizedQueue,
)
from .proxy import _Proxy
from .retries import Retries, RetryManager
from .schedule import Schedule
from .scheduler_placement import SchedulerPlacement
from .secret import _Secret
from .volume import _Volume

if TYPE_CHECKING:
    import modal._partial_function
    import modal.app
    import modal.cls
    import modal.partial_function


@dataclasses.dataclass
class _RetryContext:
    function_call_invocation_type: "api_pb2.FunctionCallInvocationType.ValueType"
    retry_policy: api_pb2.FunctionRetryPolicy
    function_call_jwt: str
    input_jwt: str
    input_id: str
    item: api_pb2.FunctionPutInputsItem
    sync_client_retries_enabled: bool


class _Invocation:
    """Internal client representation of a single-input call to a Modal Function or Generator"""

    stub: ModalClientModal

    def __init__(
        self,
        stub: ModalClientModal,
        function_call_id: str,
        client: _Client,
        retry_context: Optional[_RetryContext] = None,
    ):
        self.stub = stub
        self.client = client  # Used by the deserializer.
        self.function_call_id = function_call_id  # TODO: remove and use only input_id
        self._retry_context = retry_context

    @staticmethod
    async def create(
        function: "_Function",
        args,
        kwargs,
        *,
        client: _Client,
        function_call_invocation_type: "api_pb2.FunctionCallInvocationType.ValueType",
    ) -> "_Invocation":
        assert client.stub
        function_id = function.object_id
        item = await _create_input(args, kwargs, client, method_name=function._use_method_name)

        request = api_pb2.FunctionMapRequest(
            function_id=function_id,
            parent_input_id=current_input_id() or "",
            function_call_type=api_pb2.FUNCTION_CALL_TYPE_UNARY,
            pipelined_inputs=[item],
            function_call_invocation_type=function_call_invocation_type,
        )
        response = await retry_transient_errors(client.stub.FunctionMap, request)
        function_call_id = response.function_call_id

        if response.pipelined_inputs:
            assert len(response.pipelined_inputs) == 1
            input = response.pipelined_inputs[0]
            retry_context = _RetryContext(
                function_call_invocation_type=function_call_invocation_type,
                retry_policy=response.retry_policy,
                function_call_jwt=response.function_call_jwt,
                input_jwt=input.input_jwt,
                input_id=input.input_id,
                item=item,
                sync_client_retries_enabled=response.sync_client_retries_enabled,
            )
            return _Invocation(client.stub, function_call_id, client, retry_context)

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
        input = inputs_response.inputs[0]
        retry_context = _RetryContext(
            function_call_invocation_type=function_call_invocation_type,
            retry_policy=response.retry_policy,
            function_call_jwt=response.function_call_jwt,
            input_jwt=input.input_jwt,
            input_id=input.input_id,
            item=item,
            sync_client_retries_enabled=response.sync_client_retries_enabled,
        )
        return _Invocation(client.stub, function_call_id, client, retry_context)

    async def pop_function_call_outputs(
        self, timeout: Optional[float], clear_on_success: bool, input_jwts: Optional[list[str]] = None
    ) -> api_pb2.FunctionGetOutputsResponse:
        t0 = time.time()
        if timeout is None:
            backend_timeout = OUTPUTS_TIMEOUT
        else:
            # refresh backend call every 55s
            backend_timeout = min(OUTPUTS_TIMEOUT, timeout)

        while True:
            # always execute at least one poll for results, regardless if timeout is 0
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=self.function_call_id,
                timeout=backend_timeout,
                last_entry_id="0-0",
                clear_on_success=clear_on_success,
                requested_at=time.time(),
                input_jwts=input_jwts,
            )
            response: api_pb2.FunctionGetOutputsResponse = await retry_transient_errors(
                self.stub.FunctionGetOutputs,
                request,
                attempt_timeout=backend_timeout + ATTEMPT_TIMEOUT_GRACE_PERIOD,
            )

            if len(response.outputs) > 0:
                return response

            if timeout is not None:
                # update timeout in retry loop
                backend_timeout = min(OUTPUTS_TIMEOUT, t0 + timeout - time.time())
                if backend_timeout < 0:
                    # return the last response to check for state of num_unfinished_inputs
                    return response

    async def _retry_input(self) -> None:
        ctx = self._retry_context
        if not ctx:
            raise ValueError("Cannot retry input when _retry_context is empty.")

        item = api_pb2.FunctionRetryInputsItem(input_jwt=ctx.input_jwt, input=ctx.item.input)
        request = api_pb2.FunctionRetryInputsRequest(function_call_jwt=ctx.function_call_jwt, inputs=[item])
        await retry_transient_errors(
            self.client.stub.FunctionRetryInputs,
            request,
        )

    async def _get_single_output(self, expected_jwt: Optional[str] = None) -> Any:
        # waits indefinitely for a single result for the function, and clear the outputs buffer after
        item: api_pb2.FunctionGetOutputsItem = (
            await self.pop_function_call_outputs(
                timeout=None,
                clear_on_success=True,
                input_jwts=[expected_jwt] if expected_jwt else None,
            )
        ).outputs[0]
        return await _process_result(item.result, item.data_format, self.stub, self.client)

    async def run_function(self) -> Any:
        # Use retry logic only if retry policy is specified and
        ctx = self._retry_context
        if (
            not ctx
            or not ctx.retry_policy
            or ctx.retry_policy.retries == 0
            or ctx.function_call_invocation_type != api_pb2.FUNCTION_CALL_INVOCATION_TYPE_SYNC
            or not ctx.sync_client_retries_enabled
        ):
            return await self._get_single_output()

        # User errors including timeouts are managed by the user specified retry policy.
        user_retry_manager = RetryManager(ctx.retry_policy)

        while True:
            try:
                return await self._get_single_output(ctx.input_jwt)
            except (UserCodeException, FunctionTimeoutError) as exc:
                delay_ms = user_retry_manager.get_delay_ms()
                if delay_ms is None:
                    raise exc
                await asyncio.sleep(delay_ms / 1000)
            except InternalFailure:
                # For system failures on the server, we retry immediately,
                # and the failure does not count towards the retry policy.
                pass
            await self._retry_input()

    async def poll_function(self, timeout: Optional[float] = None):
        """Waits up to timeout for a result from a function.

        If timeout is `None`, waits indefinitely. This function is not
        cancellation-safe.
        """
        response: api_pb2.FunctionGetOutputsResponse = await self.pop_function_call_outputs(
            timeout=timeout, clear_on_success=False
        )
        if len(response.outputs) == 0 and response.num_unfinished_inputs == 0:
            # if no unfinished inputs and no outputs, then function expired
            raise OutputExpiredError()
        elif len(response.outputs) == 0:
            raise TimeoutError()

        return await _process_result(
            response.outputs[0].result, response.outputs[0].data_format, self.stub, self.client
        )

    async def run_generator(self):
        items_received = 0
        # populated when self.run_function() completes
        items_total: Union[int, None] = None
        async with aclosing(
            async_merge(
                _stream_function_call_data(self.client, self.function_call_id, variant="data_out"),
                callable_to_agen(self.run_function),
            )
        ) as streamer:
            async for item in streamer:
                if isinstance(item, api_pb2.GeneratorDone):
                    items_total = item.items_total
                else:
                    yield item
                    items_received += 1
                # The comparison avoids infinite loops if a non-deterministic generator is retried
                # and produces less data in the second run than what was already sent.
                if items_total is not None and items_received >= items_total:
                    break


# Wrapper type for api_pb2.FunctionStats
@dataclass(frozen=True)
class FunctionStats:
    """Simple data structure storing stats for a running function."""

    backlog: int
    num_total_runners: int

    def __getattr__(self, name):
        if name == "num_active_runners":
            msg = "'FunctionStats.num_active_runners' is no longer available."
            deprecation_error((2024, 6, 14), msg)
        raise AttributeError(f"'FunctionStats' object has no attribute '{name}'")


def _parse_retries(
    retries: Optional[Union[int, Retries]],
    source: str = "",
) -> Optional[api_pb2.FunctionRetryPolicy]:
    if isinstance(retries, int):
        return Retries(
            max_retries=retries,
            initial_delay=1.0,
            backoff_coefficient=1.0,
        )._to_proto()
    elif isinstance(retries, Retries):
        return retries._to_proto()
    elif retries is None:
        return None
    else:
        extra = f" on {source}" if source else ""
        msg = f"Retries parameter must be an integer or instance of modal.Retries. Found: {type(retries)}{extra}."
        raise InvalidError(msg)


@dataclass
class _FunctionSpec:
    """
    Stores information about a Function specification.
    This is used for `modal shell` to support running shells with
    the same configuration as a user-defined Function.
    """

    image: Optional[_Image]
    mounts: Sequence[_Mount]
    secrets: Sequence[_Secret]
    network_file_systems: dict[Union[str, PurePosixPath], _NetworkFileSystem]
    volumes: dict[Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]]
    # TODO(irfansharif): Somehow assert that it's the first kind, in sandboxes
    gpus: Union[GPU_T, list[GPU_T]]
    cloud: Optional[str]
    cpu: Optional[Union[float, tuple[float, float]]]
    memory: Optional[Union[int, tuple[int, int]]]
    ephemeral_disk: Optional[int]
    scheduler_placement: Optional[SchedulerPlacement]
    proxy: Optional[_Proxy]


P = typing_extensions.ParamSpec("P")
ReturnType = typing.TypeVar("ReturnType", covariant=True)
OriginalReturnType = typing.TypeVar(
    "OriginalReturnType", covariant=True
)  # differs from return type if ReturnType is coroutine


class _Function(typing.Generic[P, ReturnType, OriginalReturnType], _Object, type_prefix="fu"):
    """Functions are the basic units of serverless execution on Modal.

    Generally, you will not construct a `Function` directly. Instead, use the
    `App.function()` decorator to register your Python functions with your App.
    """

    # TODO: more type annotations
    _info: Optional[FunctionInfo]
    _serve_mounts: frozenset[_Mount]  # set at load time, only by loader
    _app: Optional["modal.app._App"] = None
    # only set for InstanceServiceFunctions and bound instance methods
    _obj: Optional["modal.cls._Obj"] = None

    # this is set in definition scope, only locally
    _webhook_config: Optional[api_pb2.WebhookConfig] = None
    _web_url: Optional[str]  # this is set on hydration

    _function_name: Optional[str]
    _is_method: bool
    _spec: Optional[_FunctionSpec] = None
    _tag: str
    # this is set to None for a "class service [function]"
    _raw_f: Optional[Callable[..., Any]]
    _build_args: dict

    _is_generator: Optional[bool] = None
    _cluster_size: Optional[int] = None

    # when this is the method of a class/object function, invocation of this function
    # should supply the method name in the FunctionInput:
    _use_method_name: str = ""

    _class_parameter_info: Optional["api_pb2.ClassParameterInfo"] = None
    _method_handle_metadata: Optional[dict[str, "api_pb2.FunctionHandleMetadata"]] = (
        None  # set for 0.67+ class service functions
    )
    _metadata: Optional[api_pb2.FunctionHandleMetadata] = None

    @staticmethod
    def from_local(
        info: FunctionInfo,
        app,
        image: _Image,
        secrets: Sequence[_Secret] = (),
        schedule: Optional[Schedule] = None,
        is_generator: bool = False,
        gpu: Union[GPU_T, list[GPU_T]] = None,
        # TODO: maybe break this out into a separate decorator for notebooks.
        mounts: Collection[_Mount] = (),
        network_file_systems: dict[Union[str, PurePosixPath], _NetworkFileSystem] = {},
        allow_cross_region_volumes: bool = False,
        volumes: dict[Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]] = {},
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
        cpu: Optional[Union[float, tuple[float, float]]] = None,
        memory: Optional[Union[int, tuple[int, int]]] = None,
        proxy: Optional[_Proxy] = None,
        retries: Optional[Union[int, Retries]] = None,
        timeout: Optional[int] = None,
        min_containers: Optional[int] = None,
        max_containers: Optional[int] = None,
        buffer_containers: Optional[int] = None,
        scaledown_window: Optional[int] = None,
        max_concurrent_inputs: Optional[int] = None,
        target_concurrent_inputs: Optional[int] = None,
        batch_max_size: Optional[int] = None,
        batch_wait_ms: Optional[int] = None,
        cloud: Optional[str] = None,
        scheduler_placement: Optional[SchedulerPlacement] = None,
        is_builder_function: bool = False,
        is_auto_snapshot: bool = False,
        enable_memory_snapshot: bool = False,
        block_network: bool = False,
        i6pn_enabled: bool = False,
        # Experimental: Clustered functions
        cluster_size: Optional[int] = None,
        max_inputs: Optional[int] = None,
        ephemeral_disk: Optional[int] = None,
        # current default: first-party, future default: main-package
        include_source: Optional[bool] = None,
        _experimental_proxy_ip: Optional[str] = None,
        _experimental_custom_scaling_factor: Optional[float] = None,
        _experimental_enable_gpu_snapshot: bool = False,
    ) -> "_Function":
        """mdmd:hidden"""
        # Needed to avoid circular imports
        from ._partial_function import _find_partial_methods_for_user_cls, _PartialFunctionFlags

        tag = info.get_tag()

        if info.raw_f:
            raw_f = info.raw_f
            assert callable(raw_f)
            if schedule is not None and not info.is_nullary():
                raise InvalidError(
                    f"Function {raw_f} has a schedule, so it needs to support being called with no arguments"
                )
        else:
            # must be a "class service function"
            assert info.user_cls
            assert not webhook_config
            assert not schedule

        explicit_mounts = mounts

        include_source_mode = get_include_source_mode(include_source)
        if include_source_mode != IncludeSourceMode.INCLUDE_NOTHING:
            entrypoint_mounts = info.get_entrypoint_mount()
        else:
            entrypoint_mounts = {}

        all_mounts = [
            _get_client_mount(),
            *explicit_mounts,
            *entrypoint_mounts.values(),
        ]

        if include_source_mode is IncludeSourceMode.INCLUDE_FIRST_PARTY and is_local():
            auto_mounts = get_sys_modules_mounts()
            # don't need to add entrypoint modules to automounts:
            for entrypoint_module in entrypoint_mounts:
                auto_mounts.pop(entrypoint_module, None)

            warn_missing_modules = set(auto_mounts.keys()) - image._added_python_source_set

            if warn_missing_modules:
                python_stringified_modules = ", ".join(f'"{mod}"' for mod in sorted(warn_missing_modules))
                deprecation_warning(
                    (2025, 2, 3),
                    (
                        'Modal will stop implicitly adding local Python modules to the Image ("automounting") in a '
                        "future update. The following modules need to be explicitly added for future "
                        "compatibility:\n"
                    )
                    + "\n".join(sorted([f"* {m}" for m in warn_missing_modules]))
                    + "\n\n"
                    + (f"e.g.:\nimage_with_source = my_image.add_local_python_source({python_stringified_modules})\n\n")
                    + "For more information, see https://modal.com/docs/guide/modal-1-0-migration",
                )
            all_mounts += auto_mounts.values()

        retry_policy = _parse_retries(
            retries, f"Function '{info.get_tag()}'" if info.raw_f else f"Class '{info.get_tag()}'"
        )

        if retry_policy is not None:
            if webhook_config is not None:
                raise InvalidError("Web endpoints do not support retries.")
            if is_generator:
                raise InvalidError("Generator functions do not support retries.")

        function_spec = _FunctionSpec(
            mounts=all_mounts,
            secrets=secrets,
            gpus=gpu,
            network_file_systems=network_file_systems,
            volumes=volumes,
            image=image,
            cloud=cloud,
            cpu=cpu,
            memory=memory,
            ephemeral_disk=ephemeral_disk,
            scheduler_placement=scheduler_placement,
            proxy=proxy,
        )

        if info.user_cls and not is_auto_snapshot:
            build_functions = _find_partial_methods_for_user_cls(info.user_cls, _PartialFunctionFlags.BUILD).items()
            for k, pf in build_functions:
                build_function = pf.raw_f
                snapshot_info = FunctionInfo(build_function, user_cls=info.user_cls)
                snapshot_function = _Function.from_local(
                    snapshot_info,
                    app=None,
                    image=image,
                    secrets=secrets,
                    gpu=gpu,
                    mounts=mounts,
                    network_file_systems=network_file_systems,
                    volumes=volumes,
                    memory=memory,
                    timeout=pf.params.build_timeout,
                    cpu=cpu,
                    ephemeral_disk=ephemeral_disk,
                    is_builder_function=True,
                    is_auto_snapshot=True,
                    scheduler_placement=scheduler_placement,
                    include_source=include_source,
                )
                image = _Image._from_args(
                    base_images={"base": image},
                    build_function=snapshot_function,
                    force_build=image.force_build or bool(pf.params.force_build),
                )

        # Note that we also do these checks in FunctionCreate; could drop them here
        if min_containers is not None and not isinstance(min_containers, int):
            raise InvalidError(f"`min_containers` must be an int, not {type(min_containers).__name__}")
        if min_containers is not None and max_containers is not None and max_containers < min_containers:
            raise InvalidError(
                f"`min_containers` ({min_containers}) cannot be greater than `max_containers` ({max_containers})"
            )
        if scaledown_window is not None and scaledown_window <= 0:
            raise InvalidError("`scaledown_window` must be > 0")

        autoscaler_settings = api_pb2.AutoscalerSettings(
            min_containers=min_containers,
            max_containers=max_containers,
            buffer_containers=buffer_containers,
            scaledown_window=scaledown_window,
        )

        if _experimental_custom_scaling_factor is not None and (
            _experimental_custom_scaling_factor < 0 or _experimental_custom_scaling_factor > 1
        ):
            raise InvalidError("`_experimental_custom_scaling_factor` must be between 0.0 and 1.0 inclusive.")

        if not cloud and not is_builder_function:
            cloud = config.get("default_cloud")

        if is_generator and webhook_config:
            if webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
                raise InvalidError(
                    """Webhooks cannot be generators. If you want a streaming response, see https://modal.com/docs/guide/streaming-endpoints
                    """
                )
            else:
                raise InvalidError("Webhooks cannot be generators")

        if info.raw_f and batch_max_size:
            func_name = info.raw_f.__name__
            if is_generator:
                raise InvalidError(f"Modal batched function {func_name} cannot return generators")
            for arg in inspect.signature(info.raw_f).parameters.values():
                if arg.default is not inspect.Parameter.empty:
                    raise InvalidError(f"Modal batched function {func_name} does not accept default arguments.")

        if max_inputs is not None:
            if not isinstance(max_inputs, int):
                raise InvalidError(f"`max_inputs` must be an int, not {type(max_inputs).__name__}")
            if max_inputs <= 0:
                raise InvalidError("`max_inputs` must be positive")
            if max_inputs > 1:
                raise InvalidError("Only `max_inputs=1` is currently supported")

        # Validate volumes
        validated_volumes = validate_volumes(volumes)
        cloud_bucket_mounts = [(k, v) for k, v in validated_volumes if isinstance(v, _CloudBucketMount)]
        validated_volumes_no_cloud_buckets = [(k, v) for k, v in validated_volumes if isinstance(v, _Volume)]

        # Validate NFS
        validated_network_file_systems = validate_network_file_systems(network_file_systems)

        # Validate image
        if image is not None and not isinstance(image, _Image):
            raise InvalidError(f"Expected modal.Image object. Got {type(image)}.")

        method_definitions: Optional[dict[str, api_pb2.MethodDefinition]] = None

        if info.user_cls:
            method_definitions = {}
            interface_methods = _find_partial_methods_for_user_cls(
                info.user_cls, _PartialFunctionFlags.interface_flags()
            )
            for method_name, partial_function in interface_methods.items():
                function_type = get_function_type(partial_function.params.is_generator)
                function_name = f"{info.user_cls.__name__}.{method_name}"
                method_schema = get_callable_schema(
                    partial_function._get_raw_f(),
                    is_web_endpoint=partial_function._is_web_endpoint(),
                    ignore_first_argument=True,
                )

                method_definition = api_pb2.MethodDefinition(
                    webhook_config=partial_function.params.webhook_config,
                    function_type=function_type,
                    function_name=function_name,
                    function_schema=method_schema,
                )
                method_definitions[method_name] = method_definition

        function_type = get_function_type(is_generator)

        def _deps(only_explicit_mounts=False) -> list[_Object]:
            deps: list[_Object] = list(secrets)
            if only_explicit_mounts:
                # TODO: this is a bit hacky, but all_mounts may differ in the container vs locally
                # We don't want the function dependencies to change, so we have this way to force it to
                # only include its declared dependencies.
                # Only objects that need interaction within a user's container actually need to be
                # included when only_explicit_mounts=True, so omitting auto mounts here
                # wouldn't be a problem as long as Mounts are "passive" and only loaded by the
                # worker runtime
                deps += list(explicit_mounts)
            else:
                deps += list(all_mounts)
            if proxy:
                deps.append(proxy)
            if image:
                deps.append(image)
            for _, nfs in validated_network_file_systems:
                deps.append(nfs)
            for _, vol in validated_volumes_no_cloud_buckets:
                deps.append(vol)
            for _, cloud_bucket_mount in cloud_bucket_mounts:
                if cloud_bucket_mount.secret:
                    deps.append(cloud_bucket_mount.secret)

            return deps

        async def _preload(self: _Function, resolver: Resolver, existing_object_id: Optional[str]):
            assert resolver.client and resolver.client.stub

            assert resolver.app_id
            req = api_pb2.FunctionPrecreateRequest(
                app_id=resolver.app_id,
                function_name=info.function_name,
                function_type=function_type,
                existing_function_id=existing_object_id or "",
                function_schema=get_callable_schema(info.raw_f, is_web_endpoint=bool(webhook_config))
                if info.raw_f
                else None,
            )
            if method_definitions:
                for method_name, method_definition in method_definitions.items():
                    req.method_definitions[method_name].CopyFrom(method_definition)
            elif webhook_config:
                req.webhook_config.CopyFrom(webhook_config)
            response = await retry_transient_errors(resolver.client.stub.FunctionPrecreate, req)
            self._hydrate(response.function_id, resolver.client, response.handle_metadata)

        async def _load(self: _Function, resolver: Resolver, existing_object_id: Optional[str]):
            assert resolver.client and resolver.client.stub
            with FunctionCreationStatus(resolver, tag) as function_creation_status:
                timeout_secs = timeout

                if app and app.is_interactive and not is_builder_function:
                    pty_info = get_pty_info(shell=False)
                else:
                    pty_info = None

                if info.is_serialized():
                    # Use cloudpickle. Used when working w/ Jupyter notebooks.
                    # serialize at _load time, not function decoration time
                    # otherwise we can't capture a surrounding class for lifetime methods etc.
                    function_serialized = info.serialized_function()
                    class_serialized = serialize(info.user_cls) if info.user_cls is not None else None
                    # Ensure that large data in global variables does not blow up the gRPC payload,
                    # which has maximum size 100 MiB. We set the limit lower for performance reasons.
                    if len(function_serialized) > 16 << 20:  # 16 MiB
                        raise InvalidError(
                            f"Function {info.raw_f} has size {len(function_serialized)} bytes when packaged. "
                            "This is larger than the maximum limit of 16 MiB. "
                            "Try reducing the size of the closure by using parameters or mounts, "
                            "not large global variables."
                        )
                    elif len(function_serialized) > 256 << 10:  # 256 KiB
                        warnings.warn(
                            f"Function {info.raw_f} has size {len(function_serialized)} bytes when packaged. "
                            "This is larger than the recommended limit of 256 KiB. "
                            "Try reducing the size of the closure by using parameters or mounts, "
                            "not large global variables."
                        )
                else:
                    function_serialized = None
                    class_serialized = None

                app_name = ""
                if app and app.name:
                    app_name = app.name

                # Relies on dicts being ordered (true as of Python 3.6).
                volume_mounts = [
                    api_pb2.VolumeMount(
                        mount_path=path,
                        volume_id=volume.object_id,
                        allow_background_commits=True,
                    )
                    for path, volume in validated_volumes_no_cloud_buckets
                ]
                loaded_mount_ids = {m.object_id for m in all_mounts} | {m.object_id for m in image._mount_layers}

                # Get object dependencies
                object_dependencies = []
                for dep in _deps(only_explicit_mounts=True):
                    if not dep.object_id:
                        raise Exception(f"Dependency {dep} isn't hydrated")
                    object_dependencies.append(api_pb2.ObjectDependency(object_id=dep.object_id))

                function_data: Optional[api_pb2.FunctionData] = None
                function_schema = (
                    get_callable_schema(info.raw_f, is_web_endpoint=bool(webhook_config)) if info.raw_f else None
                )
                # Create function remotely
                function_definition = api_pb2.Function(
                    module_name=info.module_name or "",
                    function_name=info.function_name,
                    mount_ids=loaded_mount_ids,
                    secret_ids=[secret.object_id for secret in secrets],
                    image_id=(image.object_id if image else ""),
                    definition_type=info.get_definition_type(),
                    function_serialized=function_serialized or b"",
                    class_serialized=class_serialized or b"",
                    function_type=function_type,
                    webhook_config=webhook_config,
                    autoscaler_settings=autoscaler_settings,
                    method_definitions=method_definitions,
                    method_definitions_set=True,
                    shared_volume_mounts=network_file_system_mount_protos(
                        validated_network_file_systems, allow_cross_region_volumes
                    ),
                    volume_mounts=volume_mounts,
                    proxy_id=(proxy.object_id if proxy else None),
                    retry_policy=retry_policy,
                    timeout_secs=timeout_secs or 0,
                    pty_info=pty_info,
                    cloud_provider_str=cloud if cloud else "",
                    runtime=config.get("function_runtime"),
                    runtime_debug=config.get("function_runtime_debug"),
                    runtime_perf_record=config.get("runtime_perf_record"),
                    app_name=app_name,
                    is_builder_function=is_builder_function,
                    max_concurrent_inputs=max_concurrent_inputs or 0,
                    target_concurrent_inputs=target_concurrent_inputs or 0,
                    batch_max_size=batch_max_size or 0,
                    batch_linger_ms=batch_wait_ms or 0,
                    worker_id=config.get("worker_id"),
                    is_auto_snapshot=is_auto_snapshot,
                    is_method=bool(info.user_cls) and not info.is_service_class(),
                    checkpointing_enabled=enable_memory_snapshot,
                    object_dependencies=object_dependencies,
                    block_network=block_network,
                    max_inputs=max_inputs or 0,
                    cloud_bucket_mounts=cloud_bucket_mounts_to_proto(cloud_bucket_mounts),
                    scheduler_placement=scheduler_placement.proto if scheduler_placement else None,
                    is_class=info.is_service_class(),
                    class_parameter_info=info.class_parameter_info(),
                    i6pn_enabled=i6pn_enabled,
                    schedule=schedule.proto_message if schedule is not None else None,
                    snapshot_debug=config.get("snapshot_debug"),
                    _experimental_group_size=cluster_size or 0,  # Experimental: Clustered functions
                    _experimental_concurrent_cancellations=True,
                    _experimental_proxy_ip=_experimental_proxy_ip,
                    _experimental_custom_scaling=_experimental_custom_scaling_factor is not None,
                    _experimental_enable_gpu_snapshot=_experimental_enable_gpu_snapshot,
                    # --- These are deprecated in favor of autoscaler_settings
                    warm_pool_size=min_containers or 0,
                    concurrency_limit=max_containers or 0,
                    _experimental_buffer_containers=buffer_containers or 0,
                    task_idle_timeout_secs=scaledown_window or 0,
                    # ---
                    function_schema=function_schema,
                )

                if isinstance(gpu, list):
                    function_data = api_pb2.FunctionData(
                        module_name=function_definition.module_name,
                        function_name=function_definition.function_name,
                        function_type=function_definition.function_type,
                        warm_pool_size=function_definition.warm_pool_size,
                        concurrency_limit=function_definition.concurrency_limit,
                        task_idle_timeout_secs=function_definition.task_idle_timeout_secs,
                        autoscaler_settings=function_definition.autoscaler_settings,
                        worker_id=function_definition.worker_id,
                        timeout_secs=function_definition.timeout_secs,
                        web_url=function_definition.web_url,
                        web_url_info=function_definition.web_url_info,
                        webhook_config=function_definition.webhook_config,
                        custom_domain_info=function_definition.custom_domain_info,
                        schedule=schedule.proto_message if schedule is not None else None,
                        is_class=function_definition.is_class,
                        class_parameter_info=function_definition.class_parameter_info,
                        is_method=function_definition.is_method,
                        use_function_id=function_definition.use_function_id,
                        use_method_name=function_definition.use_method_name,
                        method_definitions=function_definition.method_definitions,
                        method_definitions_set=function_definition.method_definitions_set,
                        _experimental_group_size=function_definition._experimental_group_size,
                        _experimental_buffer_containers=function_definition._experimental_buffer_containers,
                        _experimental_custom_scaling=function_definition._experimental_custom_scaling,
                        _experimental_enable_gpu_snapshot=_experimental_enable_gpu_snapshot,
                        _experimental_proxy_ip=function_definition._experimental_proxy_ip,
                        snapshot_debug=function_definition.snapshot_debug,
                        runtime_perf_record=function_definition.runtime_perf_record,
                        function_schema=function_schema,
                    )

                    ranked_functions = []
                    for rank, _gpu in enumerate(gpu):
                        function_definition_copy = api_pb2.Function()
                        function_definition_copy.CopyFrom(function_definition)

                        function_definition_copy.resources.CopyFrom(
                            convert_fn_config_to_resources_config(
                                cpu=cpu, memory=memory, gpu=_gpu, ephemeral_disk=ephemeral_disk
                            ),
                        )
                        ranked_function = api_pb2.FunctionData.RankedFunction(
                            rank=rank,
                            function=function_definition_copy,
                        )
                        ranked_functions.append(ranked_function)
                    function_data.ranked_functions.extend(ranked_functions)
                    function_definition = None  # function_definition is not used in this case
                else:
                    # TODO(irfansharif): Assert on this specific type once we get rid of python 3.9.
                    # assert isinstance(gpu, GPU_T)  # includes the case where gpu==None case
                    function_definition.resources.CopyFrom(
                        convert_fn_config_to_resources_config(
                            cpu=cpu, memory=memory, gpu=gpu, ephemeral_disk=ephemeral_disk
                        ),
                    )

                assert resolver.app_id
                assert (function_definition is None) != (function_data is None)  # xor
                request = api_pb2.FunctionCreateRequest(
                    app_id=resolver.app_id,
                    function=function_definition,
                    function_data=function_data,
                    existing_function_id=existing_object_id or "",
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
                    if exc.message and "Received :status = '413'" in exc.message:
                        raise InvalidError(f"Function {info.function_name} is too large to deploy.")
                    raise
                function_creation_status.set_response(response)
            # needed for modal.serve file watching
            serve_mounts = {m for m in all_mounts if m.is_local()}
            serve_mounts |= image._serve_mounts
            obj._serve_mounts = frozenset(serve_mounts)
            self._hydrate(response.function_id, resolver.client, response.handle_metadata)

        rep = f"Function({tag})"
        obj = _Function._from_loader(_load, rep, preload=_preload, deps=_deps)

        obj._raw_f = info.raw_f
        obj._info = info
        obj._tag = tag
        obj._app = app  # needed for CLI right now
        obj._obj = None
        obj._is_generator = is_generator
        obj._cluster_size = cluster_size
        obj._is_method = False
        obj._spec = function_spec  # needed for modal shell
        obj._webhook_config = webhook_config  # only set locally

        # Used to check whether we should rebuild a modal.Image which uses `run_function`.
        gpus: list[GPU_T] = gpu if isinstance(gpu, list) else [gpu]
        obj._build_args = dict(  # See get_build_def
            secrets=repr(secrets),
            gpu_config=repr([parse_gpu_config(_gpu) for _gpu in gpus]),
            mounts=repr(mounts),
            network_file_systems=repr(network_file_systems),
        )
        # these key are excluded if empty to avoid rebuilds on client upgrade
        if volumes:
            obj._build_args["volumes"] = repr(volumes)
        if cloud or scheduler_placement:
            obj._build_args["cloud"] = repr(cloud)
            obj._build_args["scheduler_placement"] = repr(scheduler_placement)

        return obj

    def _bind_parameters(
        self,
        obj: "modal.cls._Obj",
        options: Optional["modal.cls._ServiceOptions"],
        args: Sized,
        kwargs: dict[str, Any],
    ) -> "_Function":
        """mdmd:hidden

        Binds a class-function to a specific instance of (init params, options) or a new workspace
        """

        parent = self

        async def _load(param_bound_func: _Function, resolver: Resolver, existing_object_id: Optional[str]):
            try:
                identity = f"{parent.info.function_name} class service function"
            except Exception:
                # Can't always look up the function name that way, so fall back to generic message
                identity = "class service function for a parametrized class"
            if not parent.is_hydrated:
                if parent.app._running_app is None:
                    reason = ", because the App it is defined on is not running"
                else:
                    reason = ""
                raise ExecutionError(
                    f"The {identity} has not been hydrated with the metadata it needs to run on Modal{reason}."
                )

            assert parent._client and parent._client.stub

            if (
                parent._class_parameter_info
                and parent._class_parameter_info.format == api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PROTO
            ):
                if args:
                    # TODO(elias) - We could potentially support positional args as well, if we want to?
                    raise InvalidError(
                        "Can't use positional arguments with modal.parameter-based synthetic constructors.\n"
                        "Use (<parameter_name>=value) keyword arguments when constructing classes instead."
                    )
                schema = parent._class_parameter_info.schema
                kwargs_with_defaults = apply_defaults(kwargs, schema)
                validate_parameter_values(kwargs_with_defaults, schema)
                serialized_params = serialize_proto_params(kwargs_with_defaults)
                can_use_parent = len(parent._class_parameter_info.schema) == 0  # no parameters
            else:
                can_use_parent = len(args) + len(kwargs) == 0 and options is None
                serialized_params = serialize((args, kwargs))

            if can_use_parent:
                # We can end up here if parent wasn't hydrated when class was instantiated, but has been since.
                param_bound_func._hydrate_from_other(parent)
                return

            environment_name = _get_environment_name(None, resolver)
            assert parent is not None and parent.is_hydrated

            if options:
                volume_mounts = [
                    api_pb2.VolumeMount(
                        mount_path=path,
                        volume_id=volume.object_id,
                        allow_background_commits=True,
                    )
                    for path, volume in options.validated_volumes
                ]
                options_pb = api_pb2.FunctionOptions(
                    secret_ids=[s.object_id for s in options.secrets],
                    replace_secret_ids=bool(options.secrets),
                    resources=options.resources,
                    retry_policy=options.retry_policy,
                    concurrency_limit=options.concurrency_limit,
                    timeout_secs=options.timeout_secs,
                    task_idle_timeout_secs=options.task_idle_timeout_secs,
                    replace_volume_mounts=len(volume_mounts) > 0,
                    volume_mounts=volume_mounts,
                    target_concurrent_inputs=options.target_concurrent_inputs,
                )
            else:
                options_pb = None

            req = api_pb2.FunctionBindParamsRequest(
                function_id=parent.object_id,
                serialized_params=serialized_params,
                function_options=options_pb,
                environment_name=environment_name
                or "",  # TODO: investigate shouldn't environment name always be specified here?
            )

            response = await retry_transient_errors(parent._client.stub.FunctionBindParams, req)
            param_bound_func._hydrate(response.bound_function_id, parent._client, response.handle_metadata)

        def _deps():
            if options:
                return [v for _, v in options.validated_volumes] + list(options.secrets)
            return []

        fun: _Function = _Function._from_loader(_load, "Function(parametrized)", hydrate_lazily=True, deps=_deps)

        fun._info = self._info
        fun._obj = obj
        fun._spec = self._spec  # TODO (elias): fix - this is incorrect when using with_options
        return fun

    @live_method
    async def keep_warm(self, warm_pool_size: int) -> None:
        """Set the warm pool size for the function.

        Please exercise care when using this advanced feature!
        Setting and forgetting a warm pool on functions can lead to increased costs.

        ```python notest
        # Usage on a regular function.
        f = modal.Function.from_name("my-app", "function")
        f.keep_warm(2)

        # Usage on a parametrized function.
        Model = modal.Cls.from_name("my-app", "Model")
        Model("fine-tuned-model").keep_warm(2)  # note that this applies to the class instance, not a method
        ```
        """
        if self._is_method:
            raise InvalidError(
                textwrap.dedent(
                    """
                The `.keep_warm()` method can not be used on Modal class *methods*.

                Call `.keep_warm()` on the class *instance* instead. All methods of a class are run by the same
                container pool, and this method applies to the size of that container pool.
            """
                )
            )
        request = api_pb2.FunctionUpdateSchedulingParamsRequest(
            function_id=self.object_id, warm_pool_size_override=warm_pool_size
        )
        await retry_transient_errors(self.client.stub.FunctionUpdateSchedulingParams, request)

    @classmethod
    def _from_name(cls, app_name: str, name: str, namespace, environment_name: Optional[str]):
        # internal function lookup implementation that allows lookup of class "service functions"
        # in addition to non-class functions
        async def _load_remote(self: _Function, resolver: Resolver, existing_object_id: Optional[str]):
            assert resolver.client and resolver.client.stub
            request = api_pb2.FunctionGetRequest(
                app_name=app_name,
                object_tag=name,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver) or "",
            )
            try:
                response = await retry_transient_errors(resolver.client.stub.FunctionGet, request)
            except GRPCError as exc:
                if exc.status == Status.NOT_FOUND:
                    env_context = f" (in the '{environment_name}' environment)" if environment_name else ""
                    raise NotFoundError(
                        f"Lookup failed for Function '{name}' from the '{app_name}' app{env_context}: {exc.message}."
                    )
                else:
                    raise

            print_server_warnings(response.server_warnings)

            self._hydrate(response.function_id, resolver.client, response.handle_metadata)

        rep = f"Function.from_name({app_name}, {name})"
        return cls._from_loader(_load_remote, rep, is_another_app=True, hydrate_lazily=True)

    @classmethod
    @renamed_parameter((2024, 12, 18), "tag", "name")
    def from_name(
        cls: type["_Function"],
        app_name: str,
        name: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
    ) -> "_Function":
        """Reference a Function from a deployed App by its name.

        In contrast to `modal.Function.lookup`, this is a lazy method
        that defers hydrating the local object with metadata from
        Modal servers until the first time it is actually used.

        ```python
        f = modal.Function.from_name("other-app", "function")
        ```
        """
        if "." in name:
            class_name, method_name = name.split(".", 1)
            deprecation_warning(
                (2025, 2, 11),
                "Looking up class methods using Function.from_name will be deprecated"
                " in a future version of Modal.\nUse modal.Cls.from_name instead, e.g.\n\n"
                f'{class_name} = modal.Cls.from_name("{app_name}", "{class_name}")\n'
                f"instance = {class_name}(...)\n"
                f"instance.{method_name}.remote(...)\n",
            )

        return cls._from_name(app_name, name, namespace, environment_name)

    @staticmethod
    @renamed_parameter((2024, 12, 18), "tag", "name")
    async def lookup(
        app_name: str,
        name: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ) -> "_Function":
        """Lookup a Function from a deployed App by its name.

        DEPRECATED: This method is deprecated in favor of `modal.Function.from_name`.

        In contrast to `modal.Function.from_name`, this is an eager method
        that will hydrate the local object with metadata from Modal servers.

        ```python notest
        f = modal.Function.lookup("other-app", "function")
        ```
        """
        deprecation_warning(
            (2025, 1, 27),
            "`modal.Function.lookup` is deprecated and will be removed in a future release."
            " It can be replaced with `modal.Function.from_name`."
            "\n\nSee https://modal.com/docs/guide/modal-1-0-migration for more information.",
        )
        obj = _Function.from_name(app_name, name, namespace=namespace, environment_name=environment_name)
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client)
        await resolver.load(obj)
        return obj

    @property
    def tag(self) -> str:
        """mdmd:hidden"""
        assert self._tag
        return self._tag

    @property
    def app(self) -> "modal.app._App":
        """mdmd:hidden"""
        if self._app is None:
            raise ExecutionError("The app has not been assigned on the function at this point")

        return self._app

    @property
    def stub(self) -> "modal.app._App":
        """mdmd:hidden"""
        # Deprecated soon, only for backwards compatibility
        return self.app

    @property
    def info(self) -> FunctionInfo:
        """mdmd:hidden"""
        assert self._info
        return self._info

    @property
    def spec(self) -> _FunctionSpec:
        """mdmd:hidden"""
        assert self._spec
        return self._spec

    def _is_web_endpoint(self) -> bool:
        # only defined in definition scope/locally, and not for class methods at the moment
        return bool(self._webhook_config and self._webhook_config.type != api_pb2.WEBHOOK_TYPE_UNSPECIFIED)

    def get_build_def(self) -> str:
        """mdmd:hidden"""
        # Plaintext source and arg definition for the function, so it's part of the image
        # hash. We can't use the cloudpickle hash because it's not very stable.
        assert hasattr(self, "_raw_f") and hasattr(self, "_build_args") and self._raw_f is not None
        return f"{inspect.getsource(self._raw_f)}\n{repr(self._build_args)}"

    # Live handle methods

    def _initialize_from_empty(self):
        # Overridden concrete implementation of base class method
        self._progress = None
        self._is_generator = None
        self._cluster_size = None
        self._web_url = None
        self._function_name = None
        self._info = None
        self._serve_mounts = frozenset()
        self._metadata = None

    def _hydrate_metadata(self, metadata: Optional[Message]):
        # Overridden concrete implementation of base class method
        assert metadata and isinstance(metadata, api_pb2.FunctionHandleMetadata), (
            f"{type(metadata)} is not FunctionHandleMetadata"
        )
        self._metadata = metadata
        # TODO: replace usage of all below with direct ._metadata access
        self._is_generator = metadata.function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR
        self._web_url = metadata.web_url
        self._function_name = metadata.function_name
        self._is_method = metadata.is_method
        self._use_method_name = metadata.use_method_name
        self._class_parameter_info = metadata.class_parameter_info
        self._method_handle_metadata = dict(metadata.method_handle_metadata)
        self._definition_id = metadata.definition_id

    def _get_metadata(self):
        # Overridden concrete implementation of base class method
        assert self._function_name, f"Function name must be set before metadata can be retrieved for {self}"
        return api_pb2.FunctionHandleMetadata(
            function_name=self._function_name,
            function_type=get_function_type(self._is_generator),
            web_url=self._web_url or "",
            use_method_name=self._use_method_name,
            is_method=self._is_method,
            class_parameter_info=self._class_parameter_info,
            definition_id=self._definition_id,
            method_handle_metadata=self._method_handle_metadata,
            function_schema=self._metadata.function_schema if self._metadata else None,
        )

    def _check_no_web_url(self, fn_name: str):
        if self._web_url:
            raise InvalidError(
                f"A webhook function cannot be invoked for remote execution with `.{fn_name}`. "
                f"Invoke this function via its web url '{self._web_url}' "
                + f"or call it locally: {self._function_name}.local()"
            )

    # TODO (live_method on properties is not great, since it could be blocking the event loop from async contexts)
    @property
    @live_method
    async def web_url(self) -> Optional[str]:
        """URL of a Function running as a web endpoint."""
        # TODO If we remove the @live_method above, we may want to provide better feedback when the underlying
        # attribute is None because the object is not hydrated, rather than because it's not a web endpoint.
        return self._web_url

    @property
    async def is_generator(self) -> bool:
        """mdmd:hidden"""
        # hacky: kind of like @live_method, but not hydrating if we have the value already from local source
        # TODO(michael) use a common / lightweight method for handling unhydrated metadata properties
        if self._is_generator is not None:
            # this is set if the function or class is local
            return self._is_generator

        # not set - this is a from_name lookup - hydrate
        await self.hydrate()
        assert self._is_generator is not None  # should be set now
        return self._is_generator

    @property
    def cluster_size(self) -> int:
        """mdmd:hidden"""
        return self._cluster_size or 1

    @live_method_gen
    async def _map(
        self, input_queue: _SynchronizedQueue, order_outputs: bool, return_exceptions: bool
    ) -> AsyncGenerator[Any, None]:
        """mdmd:hidden

        Synchronicity-wrapped map implementation. To be safe against invocations of user code in
        the synchronicity thread it doesn't accept an [async]iterator, and instead takes a
          _SynchronizedQueue instance that is fed by higher level functions like .map()

        _SynchronizedQueue is used instead of asyncio.Queue so that the main thread can put
        items in the queue safely.
        """
        self._check_no_web_url("map")
        if self._is_generator:
            raise InvalidError("A generator function cannot be called with `.map(...)`.")

        assert self._function_name
        if output_mgr := _get_output_manager():
            count_update_callback = output_mgr.function_progress_callback(self._function_name, total=None)
        else:
            count_update_callback = None

        async with aclosing(
            _map_invocation(
                self,
                input_queue,
                self.client,
                order_outputs,
                return_exceptions,
                count_update_callback,
                api_pb2.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
            )
        ) as stream:
            async for item in stream:
                yield item

    async def _call_function(self, args, kwargs) -> ReturnType:
        invocation = await _Invocation.create(
            self,
            args,
            kwargs,
            client=self.client,
            function_call_invocation_type=api_pb2.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        )

        return await invocation.run_function()

    async def _call_function_nowait(
        self, args, kwargs, function_call_invocation_type: "api_pb2.FunctionCallInvocationType.ValueType"
    ) -> _Invocation:
        return await _Invocation.create(
            self, args, kwargs, client=self.client, function_call_invocation_type=function_call_invocation_type
        )

    @warn_if_generator_is_not_consumed()
    @live_method_gen
    @synchronizer.no_input_translation
    async def _call_generator(self, args, kwargs):
        invocation = await _Invocation.create(
            self,
            args,
            kwargs,
            client=self.client,
            function_call_invocation_type=api_pb2.FUNCTION_CALL_INVOCATION_TYPE_SYNC_LEGACY,
        )
        async for res in invocation.run_generator():
            yield res

    @synchronizer.no_io_translation
    async def _call_generator_nowait(self, args, kwargs):
        deprecation_warning(
            (2024, 12, 11),
            "Calling spawn on a generator function is deprecated and will soon raise an exception.",
        )
        return await _Invocation.create(
            self,
            args,
            kwargs,
            client=self.client,
            function_call_invocation_type=api_pb2.FUNCTION_CALL_INVOCATION_TYPE_ASYNC_LEGACY,
        )

    @synchronizer.no_io_translation
    @live_method
    async def remote(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        """
        Calls the function remotely, executing it with the given arguments and returning the execution's result.
        """
        # TODO: Generics/TypeVars
        self._check_no_web_url("remote")
        if self._is_generator:
            raise InvalidError(
                "A generator function cannot be called with `.remote(...)`. Use `.remote_gen(...)` instead."
            )

        return await self._call_function(args, kwargs)

    @synchronizer.no_io_translation
    @live_method_gen
    async def remote_gen(self, *args, **kwargs) -> AsyncGenerator[Any, None]:
        """
        Calls the generator remotely, executing it with the given arguments and returning the execution's result.
        """
        # TODO: Generics/TypeVars
        self._check_no_web_url("remote_gen")

        if not self._is_generator:
            raise InvalidError(
                "A non-generator function cannot be called with `.remote_gen(...)`. Use `.remote(...)` instead."
            )
        async for item in self._call_generator(args, kwargs):
            yield item

    def _is_local(self):
        return self._info is not None

    def _get_info(self) -> FunctionInfo:
        if not self._info:
            raise ExecutionError("Can't get info for a function that isn't locally defined")
        return self._info

    def _get_obj(self) -> Optional["modal.cls._Obj"]:
        if not self._is_method:
            return None
        elif not self._obj:
            raise ExecutionError("Method has no local object")
        else:
            return self._obj

    @synchronizer.nowrap
    def local(self, *args: P.args, **kwargs: P.kwargs) -> OriginalReturnType:
        """
        Calls the function locally, executing it with the given arguments and returning the execution's result.

        The function will execute in the same environment as the caller, just like calling the underlying function
        directly in Python. In particular, only secrets available in the caller environment will be available
        through environment variables.
        """
        # TODO(erikbern): it would be nice to remove the nowrap thing, but right now that would cause
        # "user code" to run on the synchronicity thread, which seems bad
        if not self._is_local():
            msg = (
                "The definition for this function is missing here so it is not possible to invoke it locally. "
                "If this function was retrieved via `Function.lookup` you need to use `.remote()`."
            )
            raise ExecutionError(msg)

        info = self._get_info()
        if not info.raw_f:
            # Here if calling .local on a service function itself which should never happen
            # TODO: check if we end up here in a container for a serialized function?
            raise ExecutionError("Can't call .local on service function")

        if is_local() and self.spec.volumes or self.spec.network_file_systems:
            warnings.warn(
                f"The {info.function_name} function is executing locally "
                + "and will not have access to the mounted Volume or NetworkFileSystem data"
            )

        obj: Optional["modal.cls._Obj"] = self._get_obj()

        if not obj:
            fun = info.raw_f
            return fun(*args, **kwargs)
        else:
            # This is a method on a class, so bind the self to the function
            user_cls_instance = obj._cached_user_cls_instance()
            fun = info.raw_f.__get__(user_cls_instance)

            # TODO: replace implicit local enter/exit with a context manager
            if is_async(info.raw_f):
                # We want to run __aenter__ and fun in the same coroutine
                async def coro():
                    await obj._aenter()
                    return await fun(*args, **kwargs)

                return coro()  # type: ignore
            else:
                obj._enter()
                return fun(*args, **kwargs)

    @synchronizer.no_input_translation
    @live_method
    async def _experimental_spawn(self, *args: P.args, **kwargs: P.kwargs) -> "_FunctionCall[ReturnType]":
        """[Experimental] Calls the function with the given arguments, without waiting for the results.

        This experimental version of the spawn method allows up to 1 million inputs to be spawned.

        Returns a `modal.FunctionCall` object, that can later be polled or
        waited for using `.get(timeout=...)`.
        Conceptually similar to `multiprocessing.pool.apply_async`, or a Future/Promise in other contexts.
        """
        self._check_no_web_url("_experimental_spawn")
        if self._is_generator:
            invocation = await self._call_generator_nowait(args, kwargs)
        else:
            invocation = await self._call_function_nowait(
                args, kwargs, function_call_invocation_type=api_pb2.FUNCTION_CALL_INVOCATION_TYPE_ASYNC
            )

        fc: _FunctionCall[ReturnType] = _FunctionCall._new_hydrated(
            invocation.function_call_id, invocation.client, None
        )
        fc._is_generator = self._is_generator if self._is_generator else False
        return fc

    @synchronizer.no_input_translation
    @live_method
    async def spawn(self, *args: P.args, **kwargs: P.kwargs) -> "_FunctionCall[ReturnType]":
        """Calls the function with the given arguments, without waiting for the results.

        Returns a `modal.FunctionCall` object, that can later be polled or
        waited for using `.get(timeout=...)`.
        Conceptually similar to `multiprocessing.pool.apply_async`, or a Future/Promise in other contexts.
        """
        self._check_no_web_url("spawn")
        if self._is_generator:
            invocation = await self._call_generator_nowait(args, kwargs)
        else:
            invocation = await self._call_function_nowait(
                args, kwargs, api_pb2.FUNCTION_CALL_INVOCATION_TYPE_ASYNC_LEGACY
            )

        fc: _FunctionCall[ReturnType] = _FunctionCall._new_hydrated(
            invocation.function_call_id, invocation.client, None
        )
        fc._is_generator = self._is_generator if self._is_generator else False
        return fc

    def get_raw_f(self) -> Callable[..., Any]:
        """Return the inner Python object wrapped by this Modal Function."""
        assert self._raw_f is not None
        return self._raw_f

    @live_method
    async def get_current_stats(self) -> FunctionStats:
        """Return a `FunctionStats` object describing the current function's queue and runner counts."""
        resp = await retry_transient_errors(
            self.client.stub.FunctionGetCurrentStats,
            api_pb2.FunctionGetCurrentStatsRequest(function_id=self.object_id),
            total_timeout=10.0,
        )
        return FunctionStats(backlog=resp.backlog, num_total_runners=resp.num_total_tasks)

    @live_method
    async def _get_schema(self) -> api_pb2.FunctionSchema:
        """Returns recorded schema for function, internal use only for now"""
        assert self._metadata
        return self._metadata.function_schema

    # A bit hacky - but the map-style functions need to not be synchronicity-wrapped
    # in order to not execute their input iterators on the synchronicity event loop.
    # We still need to wrap them using MethodWithAio to maintain a synchronicity-like
    # api with `.aio` and get working type-stubs and reference docs generation:
    map = MethodWithAio(_map_sync, _map_async, synchronizer)
    starmap = MethodWithAio(_starmap_sync, _starmap_async, synchronizer)
    for_each = MethodWithAio(_for_each_sync, _for_each_async, synchronizer)


class _FunctionCall(typing.Generic[ReturnType], _Object, type_prefix="fc"):
    """A reference to an executed function call.

    Constructed using `.spawn(...)` on a Modal function with the same
    arguments that a function normally takes. Acts as a reference to
    an ongoing function call that can be passed around and used to
    poll or fetch function results at some later time.

    Conceptually similar to a Future/Promise/AsyncResult in other contexts and languages.
    """

    _is_generator: bool = False

    def _invocation(self):
        return _Invocation(self.client.stub, self.object_id, self.client)

    async def get(self, timeout: Optional[float] = None) -> ReturnType:
        """Get the result of the function call.

        This function waits indefinitely by default. It takes an optional
        `timeout` argument that specifies the maximum number of seconds to wait,
        which can be set to `0` to poll for an output immediately.

        The returned coroutine is not cancellation-safe.
        """

        if self._is_generator:
            raise Exception("Cannot get the result of a generator function call. Use `get_gen` instead.")

        return await self._invocation().poll_function(timeout=timeout)

    async def get_gen(self) -> AsyncGenerator[Any, None]:
        """
        Calls the generator remotely, executing it with the given arguments and returning the execution's result.
        """
        if not self._is_generator:
            raise Exception("Cannot iterate over a non-generator function call. Use `get` instead.")

        async for res in self._invocation().run_generator():
            yield res

    async def get_call_graph(self) -> list[InputInfo]:
        """Returns a structure representing the call graph from a given root
        call ID, along with the status of execution for each node.

        See [`modal.call_graph`](/docs/reference/modal.call_graph) reference page
        for documentation on the structure of the returned `InputInfo` items.
        """
        assert self._client and self._client.stub
        request = api_pb2.FunctionGetCallGraphRequest(function_call_id=self.object_id)
        response = await retry_transient_errors(self._client.stub.FunctionGetCallGraph, request)
        return _reconstruct_call_graph(response)

    async def cancel(
        self,
        # if true, containers running the inputs are forcibly terminated
        terminate_containers: bool = False,
    ):
        """Cancels the function call, which will stop its execution and mark its inputs as
        [`TERMINATED`](/docs/reference/modal.call_graph#modalcall_graphinputstatus).

        If `terminate_containers=True` - the containers running the cancelled inputs are all terminated
        causing any non-cancelled inputs on those containers to be rescheduled in new containers.
        """
        request = api_pb2.FunctionCallCancelRequest(
            function_call_id=self.object_id, terminate_containers=terminate_containers
        )
        assert self._client and self._client.stub
        await retry_transient_errors(self._client.stub.FunctionCallCancel, request)

    @staticmethod
    async def from_id(
        function_call_id: str, client: Optional[_Client] = None, is_generator: bool = False
    ) -> "_FunctionCall[Any]":
        """Instantiate a FunctionCall object from an existing ID.

        Examples:

        ```python notest
        # Spawn a FunctionCall and keep track of its object ID
        fc = my_func.spawn()
        fc_id = fc.object_id

        # Later, use the ID to re-instantiate the FunctionCall object
        fc = _FunctionCall.from_id(fc_id)
        result = fc.get()
        ```

        Note that it's only necessary to re-instantiate the `FunctionCall` with this method
        if you no longer have access to the original object returned from `Function.spawn`.

        """
        if client is None:
            client = await _Client.from_env()

        fc: _FunctionCall[Any] = _FunctionCall._new_hydrated(function_call_id, client, None)
        fc._is_generator = is_generator
        return fc

    @staticmethod
    async def gather(*function_calls: "_FunctionCall[Any]") -> list[Any]:
        """Wait until all Modal FunctionCall objects have results before returning.

        Accepts a variable number of `FunctionCall` objects, as returned by `Function.spawn()`.

        Returns a list of results from each FunctionCall, or raises an exception
        from the first failing function call.

        Examples:

        ```python notest
        fc1 = slow_func_1.spawn()
        fc2 = slow_func_2.spawn()

        result_1, result_2 = modal.FunctionCall.gather(fc1, fc2)
        ```

        *Added in v0.73.69*: This method replaces the deprecated `modal.functions.gather` function.
        """
        try:
            return await TaskContext.gather(*[fc.get() for fc in function_calls])
        except Exception as exc:
            # TODO: kill all running function calls
            raise exc


async def _gather(*function_calls: _FunctionCall[ReturnType]) -> typing.Sequence[ReturnType]:
    """Deprecated: Please use `modal.FunctionCall.gather()` instead."""
    deprecation_warning(
        (2025, 2, 24),
        "`modal.functions.gather()` is deprecated; please use `modal.FunctionCall.gather()` instead.",
    )
    return await _FunctionCall.gather(*function_calls)
