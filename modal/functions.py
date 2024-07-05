# Copyright Modal Labs 2023
import asyncio
import inspect
import textwrap
import time
import warnings
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Type,
    Union,
)

from aiostream import stream
from google.protobuf.message import Message
from grpclib import GRPCError, Status
from synchronicity.combined_types import MethodWithAio

from modal._output import FunctionCreationStatus
from modal_proto import api_grpc, api_pb2

from ._location import parse_cloud_provider
from ._output import OutputManager
from ._pty import get_pty_info
from ._resolver import Resolver
from ._resources import convert_fn_config_to_resources_config
from ._serialization import serialize
from ._utils.async_utils import (
    TaskContext,
    synchronize_api,
    synchronizer,
    warn_if_generator_is_not_consumed,
)
from ._utils.function_utils import (
    ATTEMPT_TIMEOUT_GRACE_PERIOD,
    OUTPUTS_TIMEOUT,
    FunctionInfo,
    _create_input,
    _process_result,
    _stream_function_call_data,
    is_async,
)
from ._utils.grpc_utils import retry_transient_errors
from ._utils.mount_utils import validate_mount_points, validate_volumes
from .call_graph import InputInfo, _reconstruct_call_graph
from .client import _Client
from .cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from .config import config
from .exception import (
    ExecutionError,
    InvalidError,
    NotFoundError,
    OutputExpiredError,
    deprecation_error,
    deprecation_warning,
)
from .execution_context import current_input_id, is_local
from .gpu import GPU_T, parse_gpu_config
from .image import _Image
from .mount import _get_client_mount, _Mount, get_auto_mounts
from .network_file_system import _NetworkFileSystem, network_file_system_mount_protos
from .object import _get_environment_name, _Object, live_method, live_method_gen
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
from .retries import Retries
from .schedule import Schedule
from .scheduler_placement import SchedulerPlacement
from .secret import _Secret
from .volume import _Volume

if TYPE_CHECKING:
    import modal.app
    import modal.cls
    import modal.partial_function


class _Invocation:
    """Internal client representation of a single-input call to a Modal Function or Generator"""

    stub: api_grpc.ModalClientStub

    def __init__(self, stub: api_grpc.ModalClientStub, function_call_id: str, client: _Client):
        self.stub = stub
        self.client = client  # Used by the deserializer.
        self.function_call_id = function_call_id  # TODO: remove and use only input_id

    @staticmethod
    async def create(function: "_Function", args, kwargs, *, client: _Client) -> "_Invocation":
        assert client.stub
        function_id = function._invocation_function_id()
        item = await _create_input(args, kwargs, client, method_name=function._use_method_name)

        request = api_pb2.FunctionMapRequest(
            function_id=function_id,
            parent_input_id=current_input_id() or "",
            function_call_type=api_pb2.FUNCTION_CALL_TYPE_UNARY,
            pipelined_inputs=[item],
        )
        response = await retry_transient_errors(client.stub.FunctionMap, request)
        function_call_id = response.function_call_id

        if response.pipelined_inputs:
            return _Invocation(client.stub, function_call_id, client)

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
    ) -> api_pb2.FunctionGetOutputsResponse:
        t0 = time.time()
        if timeout is None:
            backend_timeout = OUTPUTS_TIMEOUT
        else:
            backend_timeout = min(OUTPUTS_TIMEOUT, timeout)  # refresh backend call every 55s

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
                return response

            if timeout is not None:
                # update timeout in retry loop
                backend_timeout = min(OUTPUTS_TIMEOUT, t0 + timeout - time.time())
                if backend_timeout < 0:
                    # return the last response to check for state of num_unfinished_inputs
                    return response

    async def run_function(self) -> Any:
        # waits indefinitely for a single result for the function, and clear the outputs buffer after
        item: api_pb2.FunctionGetOutputsItem = (
            await self.pop_function_call_outputs(timeout=None, clear_on_success=True)
        ).outputs[0]
        assert not item.result.gen_status
        return await _process_result(item.result, item.data_format, self.stub, self.client)

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
        data_stream = _stream_function_call_data(self.client, self.function_call_id, variant="data_out")
        combined_stream = stream.merge(data_stream, stream.call(self.run_function))  # type: ignore

        items_received = 0
        items_total: Union[int, None] = None  # populated when self.run_function() completes
        async with combined_stream.stream() as streamer:
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
            msg = (
                "'FunctionStats.num_active_runners' is deprecated."
                " It currently always has a value of 0,"
                " but it will be removed in a future release."
            )
            deprecation_warning((2024, 6, 14), msg)
            return 0
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
    network_file_systems: Dict[Union[str, PurePosixPath], _NetworkFileSystem]
    volumes: Dict[Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]]
    gpu: GPU_T
    cloud: Optional[str]
    cpu: Optional[float]
    memory: Optional[Union[int, Tuple[int, int]]]
    ephemeral_disk: Optional[int]
    scheduler_placement: Optional[SchedulerPlacement]


class _Function(_Object, type_prefix="fu"):
    """Functions are the basic units of serverless execution on Modal.

    Generally, you will not construct a `Function` directly. Instead, use the
    `@app.function()` decorator on the `App` object (formerly called "Stub")
    for your application.
    """

    # TODO: more type annotations
    _info: Optional[FunctionInfo]
    _all_mounts: Collection[_Mount]
    _app: Optional["modal.app._App"] = None
    _obj: Optional["modal.cls._Obj"] = None  # only set for InstanceServiceFunctions and bound instance methods
    _web_url: Optional[str]
    _function_name: Optional[str]
    _is_method: bool
    _spec: Optional[_FunctionSpec] = None
    _tag: str
    _raw_f: Callable[..., Any]
    _build_args: dict
    _can_use_base_function: bool = False  # whether we need to call FunctionBindParams
    _is_generator: Optional[bool] = None

    # when this is the method of a class/object function, invocation of this function
    # should be using another function id and supply the method name in the FunctionInput:
    _use_function_id: str  # The function to invoke
    _use_method_name: str = ""

    # TODO (elias): remove _parent. In case of instance functions, and methods bound on those,
    #  this references the parent class-function and is used to infer the client for lazy-loaded methods
    _parent: Optional["_Function"] = None

    def _bind_method(
        self,
        user_cls,
        method_name: str,
        partial_function: "modal.partial_function._PartialFunction",
    ):
        """mdmd:hidden

        Creates a function placeholder function that binds a specific method name to
        this function for use when invoking the function.

        Should only be used on "class service functions". For "instance service functions",
        we don't create an actual backend function, and instead do client-side "fake-hydration"
        only, see _bind_instance_method.

        """
        class_service_function = self
        assert class_service_function._info  # has to be a local function to be able to "bind" it
        assert not class_service_function._is_method  # should not be used on an already bound method placeholder
        assert not class_service_function._obj  # should only be used on base function / class service function
        full_name = f"{user_cls.__name__}.{method_name}"

        if partial_function.is_generator:
            function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
        else:
            function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

        async def _load(method_bound_function: "_Function", resolver: Resolver, existing_object_id: Optional[str]):
            function_definition = api_pb2.Function(
                function_name=full_name,
                webhook_config=partial_function.webhook_config,
                function_type=function_type,
                is_method=True,
                use_function_id=class_service_function.object_id,
                use_method_name=method_name,
            )
            assert resolver.app_id
            request = api_pb2.FunctionCreateRequest(
                app_id=resolver.app_id,
                function=function_definition,
                #  method_bound_function.object_id usually gets set by preload
                existing_function_id=existing_object_id or method_bound_function.object_id or "",
            )
            assert resolver.client.stub is not None  # client should be connected when load is called
            with FunctionCreationStatus(resolver, full_name) as function_creation_status:
                response = await resolver.client.stub.FunctionCreate(request)
                method_bound_function._hydrate(
                    response.function_id,
                    resolver.client,
                    response.handle_metadata,
                )
                function_creation_status.set_response(response)

        async def _preload(method_bound_function: "_Function", resolver: Resolver, existing_object_id: Optional[str]):
            if class_service_function._use_method_name:
                raise ExecutionError(f"Can't bind method to already bound {class_service_function}")
            assert resolver.app_id
            req = api_pb2.FunctionPrecreateRequest(
                app_id=resolver.app_id,
                function_name=full_name,
                function_type=function_type,
                webhook_config=partial_function.webhook_config,
                use_function_id=class_service_function.object_id,
                use_method_name=method_name,
                existing_function_id=existing_object_id or "",
            )
            assert resolver.client.stub  # client should be connected at this point
            response = await retry_transient_errors(resolver.client.stub.FunctionPrecreate, req)
            method_bound_function._hydrate(response.function_id, resolver.client, response.handle_metadata)

        def _deps():
            return [class_service_function]

        rep = f"Method({full_name})"

        fun = _Function._from_loader(_load, rep, preload=_preload, deps=_deps)
        fun._tag = full_name
        fun._raw_f = partial_function.raw_f
        fun._info = FunctionInfo(
            partial_function.raw_f, cls=user_cls, serialized=class_service_function.info.is_serialized()
        )  # needed for .local()
        fun._use_method_name = method_name
        fun._app = class_service_function._app
        fun._is_generator = partial_function.is_generator
        fun._all_mounts = class_service_function._all_mounts
        fun._spec = class_service_function._spec
        fun._is_method = True
        return fun

    def _bind_instance_method(self, class_bound_method: "_Function"):
        """mdmd:hidden

        Binds an "instance service function" to a specific method.
        This "dummy" _Function gets no unique object_id and isn't backend-backed at the moment, since all
        it does it forward invocations to the underlying instance_service_function with the specified method,
        and we don't support web_config for parameterized methods at the moment.
        """
        # TODO(elias): refactor to not use `_from_loader()` as a crutch for lazy-loading the
        #   underlying instance_service_function. It's currently used in order to take advantage
        #   of resolver logic and get "chained" resolution of lazy loads, even though this thin
        #   object itself doesn't need any "loading"
        instance_service_function = self
        assert instance_service_function._obj
        method_name = class_bound_method._use_method_name
        full_function_name = f"{class_bound_method._function_name}[parameterized]"

        def hydrate_from_instance_service_function(method_placeholder_fun):
            method_placeholder_fun._hydrate_from_other(instance_service_function)
            method_placeholder_fun._obj = instance_service_function._obj
            method_placeholder_fun._web_url = (
                class_bound_method._web_url
            )  # TODO: this shouldn't be set when actual parameters are used
            method_placeholder_fun._function_name = full_function_name
            method_placeholder_fun._is_generator = class_bound_method._is_generator
            method_placeholder_fun._use_method_name = method_name
            method_placeholder_fun._use_function_id = instance_service_function.object_id
            method_placeholder_fun._is_method = True

        async def _load(fun: "_Function", resolver: Resolver, existing_object_id: Optional[str]):
            # there is currently no actual loading logic executed to create each method on
            # the *parameterized* instance of a class - it uses the parameter-bound service-function
            # for the instance. This load method just makes sure to set all attributes after the
            # `instance_service_function` has been loaded (it's in the `_deps`)
            hydrate_from_instance_service_function(fun)

        def _deps():
            if instance_service_function.is_hydrated:
                # without this check, the common instance_service_function will be reloaded by all methods
                # TODO(elias): Investigate if we can fix this multi-loader in the resolver - feels like a bug?
                return []
            return [instance_service_function]

        rep = f"Method({full_function_name})"

        fun = _Function._from_loader(
            _load,
            rep,
            deps=_deps,
            hydrate_lazily=True,
        )
        if instance_service_function.is_hydrated:
            # Eager hydration (skip load) if the instance service function is already loaded
            hydrate_from_instance_service_function(fun)

        fun._info = class_bound_method._info
        fun._obj = instance_service_function._obj
        fun._is_method = True
        fun._parent = instance_service_function._parent
        fun._app = class_bound_method._app
        fun._all_mounts = class_bound_method._all_mounts  # TODO: only used for mount-watching/modal serve
        fun._spec = class_bound_method._spec
        return fun

    @staticmethod
    def from_args(
        info: FunctionInfo,
        app,
        image: _Image,
        secret: Optional[_Secret] = None,
        secrets: Sequence[_Secret] = (),
        schedule: Optional[Schedule] = None,
        is_generator=False,
        gpu: GPU_T = None,
        # TODO: maybe break this out into a separate decorator for notebooks.
        mounts: Collection[_Mount] = (),
        network_file_systems: Dict[Union[str, PurePosixPath], _NetworkFileSystem] = {},
        allow_cross_region_volumes: bool = False,
        volumes: Dict[Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]] = {},
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
        memory: Optional[Union[int, Tuple[int, int]]] = None,
        proxy: Optional[_Proxy] = None,
        retries: Optional[Union[int, Retries]] = None,
        timeout: Optional[int] = None,
        concurrency_limit: Optional[int] = None,
        allow_concurrent_inputs: Optional[int] = None,
        container_idle_timeout: Optional[int] = None,
        cpu: Optional[float] = None,
        keep_warm: Optional[int] = None,  # keep_warm=True is equivalent to keep_warm=1
        cloud: Optional[str] = None,
        _experimental_boost: bool = False,
        scheduler_placement: Optional[SchedulerPlacement] = None,
        is_builder_function: bool = False,
        is_auto_snapshot: bool = False,
        enable_memory_snapshot: bool = False,
        checkpointing_enabled: Optional[bool] = None,
        allow_background_volume_commits: Optional[bool] = None,
        block_network: bool = False,
        max_inputs: Optional[int] = None,
        ephemeral_disk: Optional[int] = None,
    ) -> None:
        """mdmd:hidden"""
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
            assert info.cls
            assert not webhook_config
            assert not schedule

        if secret is not None:
            deprecation_error(
                (2024, 1, 31),
                "The singular `secret` parameter is deprecated. Pass a list to `secrets` instead.",
            )

        if checkpointing_enabled is not None:
            deprecation_warning(
                (2024, 3, 4),
                "The argument `checkpointing_enabled` is now deprecated. Use `enable_memory_snapshot` instead.",
            )
            enable_memory_snapshot = checkpointing_enabled

        if allow_background_volume_commits is False:
            deprecation_warning(
                (2024, 5, 13),
                "Disabling volume background commits is now deprecated. Set _allow_background_volume_commits=True.",
            )
        elif allow_background_volume_commits is None:
            allow_background_volume_commits = True

        explicit_mounts = mounts

        if is_local():
            entrypoint_mounts = info.get_entrypoint_mount()
            all_mounts = [
                _get_client_mount(),
                *explicit_mounts,
                *entrypoint_mounts,
            ]

            if config.get("automount"):
                all_mounts += get_auto_mounts()
        else:
            # skip any mount introspection/logic inside containers, since the function
            # should already be hydrated
            # TODO: maybe the entire constructor should be exited early if not local?
            all_mounts = []

        retry_policy = _parse_retries(
            retries, f"Function '{info.get_tag()}'" if info.raw_f else f"Class '{info.get_tag()}'"
        )

        if webhook_config is not None and retry_policy is not None:
            raise InvalidError(
                "Web endpoints do not support retries.",
            )

        if is_generator and retry_policy is not None:
            deprecation_warning(
                (2024, 6, 25),
                "Retries for generator functions are deprecated and will soon be removed.",
            )

        gpu_config = parse_gpu_config(gpu)

        if proxy:
            # HACK: remove this once we stop using ssh tunnels for this.
            if image:
                image = image.apt_install("autossh")

        function_spec = _FunctionSpec(
            mounts=all_mounts,
            secrets=secrets,
            gpu=gpu,
            network_file_systems=network_file_systems,
            volumes=volumes,
            image=image,
            cloud=cloud,
            cpu=cpu,
            memory=memory,
            ephemeral_disk=ephemeral_disk,
            scheduler_placement=scheduler_placement,
        )

        if info.cls and not is_auto_snapshot:
            # Needed to avoid circular imports
            from .partial_function import _find_callables_for_cls, _PartialFunctionFlags

            build_functions = list(_find_callables_for_cls(info.cls, _PartialFunctionFlags.BUILD).values())
            for build_function in build_functions:
                snapshot_info = FunctionInfo(build_function, cls=info.cls)
                snapshot_function = _Function.from_args(
                    snapshot_info,
                    app=None,
                    image=image,
                    secrets=secrets,
                    gpu=gpu,
                    mounts=mounts,
                    network_file_systems=network_file_systems,
                    volumes=volumes,
                    memory=memory,
                    timeout=86400,  # TODO: make this an argument to `@build()`
                    cpu=cpu,
                    ephemeral_disk=ephemeral_disk,
                    is_builder_function=True,
                    is_auto_snapshot=True,
                    scheduler_placement=scheduler_placement,
                )
                image = _Image._from_args(
                    base_images={"base": image},
                    build_function=snapshot_function,
                    force_build=image.force_build,
                )

        if keep_warm is not None and not isinstance(keep_warm, int):
            raise TypeError(f"`keep_warm` must be an int or bool, not {type(keep_warm).__name__}")

        if (keep_warm is not None) and (concurrency_limit is not None) and concurrency_limit < keep_warm:
            raise InvalidError(
                f"Function `{info.function_name}` has `{concurrency_limit=}`, "
                f"strictly less than its `{keep_warm=}` parameter."
            )

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

        if container_idle_timeout is not None and container_idle_timeout <= 0:
            raise InvalidError("`container_idle_timeout` must be > 0")

        # Validate volumes
        validated_volumes = validate_volumes(volumes)
        cloud_bucket_mounts = [(k, v) for k, v in validated_volumes if isinstance(v, _CloudBucketMount)]
        validated_volumes = [(k, v) for k, v in validated_volumes if isinstance(v, _Volume)]

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
            for _, vol in validated_volumes:
                deps.append(vol)
            for _, cloud_bucket_mount in cloud_bucket_mounts:
                if cloud_bucket_mount.secret:
                    deps.append(cloud_bucket_mount.secret)

            return deps

        async def _preload(self: _Function, resolver: Resolver, existing_object_id: Optional[str]):
            assert resolver.client and resolver.client.stub
            if is_generator:
                function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
            else:
                function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

            assert resolver.app_id
            req = api_pb2.FunctionPrecreateRequest(
                app_id=resolver.app_id,
                function_name=info.function_name,
                function_type=function_type,
                webhook_config=webhook_config,
                existing_function_id=existing_object_id or "",
            )
            response = await retry_transient_errors(resolver.client.stub.FunctionPrecreate, req)
            self._hydrate(response.function_id, resolver.client, response.handle_metadata)

        async def _load(self: _Function, resolver: Resolver, existing_object_id: Optional[str]):
            assert resolver.client and resolver.client.stub
            with FunctionCreationStatus(resolver, tag) as function_creation_status:
                if is_generator:
                    function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
                else:
                    function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

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
                    class_serialized = serialize(info.cls) if info.cls is not None else None
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
                        allow_background_commits=bool(allow_background_volume_commits),
                    )
                    for path, volume in validated_volumes
                ]
                loaded_mount_ids = {m.object_id for m in all_mounts}

                # Get object dependencies
                object_dependencies = []
                for dep in _deps(only_explicit_mounts=True):
                    if not dep.object_id:
                        raise Exception(f"Dependency {dep} isn't hydrated")
                    object_dependencies.append(api_pb2.ObjectDependency(object_id=dep.object_id))

                # Create function remotely
                function_definition = api_pb2.Function(
                    module_name=info.module_name or "",
                    function_name=info.function_name,
                    mount_ids=loaded_mount_ids,
                    secret_ids=[secret.object_id for secret in secrets],
                    image_id=(image.object_id if image else ""),
                    definition_type=info.definition_type,
                    function_serialized=function_serialized or b"",
                    class_serialized=class_serialized or b"",
                    function_type=function_type,
                    resources=convert_fn_config_to_resources_config(
                        cpu=cpu, memory=memory, gpu=gpu, ephemeral_disk=ephemeral_disk
                    ),
                    webhook_config=webhook_config,
                    shared_volume_mounts=network_file_system_mount_protos(
                        validated_network_file_systems, allow_cross_region_volumes
                    ),
                    volume_mounts=volume_mounts,
                    proxy_id=(proxy.object_id if proxy else None),
                    retry_policy=retry_policy,
                    timeout_secs=timeout_secs or 0,
                    task_idle_timeout_secs=container_idle_timeout or 0,
                    concurrency_limit=concurrency_limit or 0,
                    pty_info=pty_info,
                    cloud_provider=cloud_provider,
                    warm_pool_size=keep_warm or 0,
                    runtime=config.get("function_runtime"),
                    runtime_debug=config.get("function_runtime_debug"),
                    app_name=app_name,
                    is_builder_function=is_builder_function,
                    allow_concurrent_inputs=allow_concurrent_inputs or 0,
                    worker_id=config.get("worker_id"),
                    is_auto_snapshot=is_auto_snapshot,
                    is_method=bool(info.cls) and not info.is_service_class(),
                    checkpointing_enabled=enable_memory_snapshot,
                    is_checkpointing_function=False,
                    object_dependencies=object_dependencies,
                    block_network=block_network,
                    max_inputs=max_inputs or 0,
                    cloud_bucket_mounts=cloud_bucket_mounts_to_proto(cloud_bucket_mounts),
                    _experimental_boost=_experimental_boost,
                    scheduler_placement=scheduler_placement.proto if scheduler_placement else None,
                    is_class=info.is_service_class(),
                )
                assert resolver.app_id
                request = api_pb2.FunctionCreateRequest(
                    app_id=resolver.app_id,
                    function=function_definition,
                    schedule=schedule.proto_message if schedule is not None else None,
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

            self._hydrate(response.function_id, resolver.client, response.handle_metadata)

        rep = f"Function({tag})"
        obj = _Function._from_loader(_load, rep, preload=_preload, deps=_deps)

        obj._raw_f = info.raw_f
        obj._info = info
        obj._tag = tag
        obj._all_mounts = all_mounts  # needed for modal.serve file watching
        obj._app = app  # needed for CLI right now
        obj._obj = None
        obj._is_generator = is_generator
        obj._is_method = False
        obj._spec = function_spec  # needed for modal shell

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

    def _bind_parameters(
        self,
        obj: "modal.cls._Obj",
        from_other_workspace: bool,
        options: Optional[api_pb2.FunctionOptions],
        args: Sized,
        kwargs: Dict[str, Any],
    ) -> "_Function":
        """mdmd:hidden

        Binds a class-function to a specific instance of (init params, options) or a new workspace
        """

        async def _load(self: _Function, resolver: Resolver, existing_object_id: Optional[str]):
            if self._parent is None:
                raise ExecutionError("Can't find the parent class' service function")
            try:
                identity = f"{self._parent.info.function_name} class service function"
            except Exception:
                # Can't always look up the function name that way, so fall back to generic message
                identity = "class service function for a parameterized class"
            if not self._parent.is_hydrated:
                if self._parent.app._running_app is None:
                    reason = ", because the App it is defined on is not running."
                else:
                    reason = ""
                raise ExecutionError(
                    f"The {identity} has not been hydrated with the metadata it needs to run on Modal{reason}."
                )
            assert self._parent._client.stub
            serialized_params = serialize((args, kwargs))
            environment_name = _get_environment_name(None, resolver)
            assert self._parent is not None
            req = api_pb2.FunctionBindParamsRequest(
                function_id=self._parent._object_id,
                serialized_params=serialized_params,
                function_options=options,
                environment_name=environment_name
                or "",  # TODO: investigate shouldn't environment name always be specified here?
            )

            response = await retry_transient_errors(self._parent._client.stub.FunctionBindParams, req)
            self._hydrate(response.bound_function_id, self._parent._client, response.handle_metadata)

        fun: _Function = _Function._from_loader(_load, "Function(parametrized)", hydrate_lazily=True)

        # In some cases, reuse the base function, i.e. not create new clones of each method or the "service function"
        fun._can_use_base_function = len(args) + len(kwargs) == 0 and not from_other_workspace and options is None
        if fun._can_use_base_function and self.is_hydrated:
            # Edge case that lets us hydrate all objects right away
            # if the instance didn't use explicit constructor arguments
            fun._hydrate_from_other(self)

        fun._info = self._info
        fun._obj = obj
        fun._parent = self
        return fun

    @live_method
    async def keep_warm(self, warm_pool_size: int) -> None:
        """Set the warm pool size for the function.

        Please exercise care when using this advanced feature!
        Setting and forgetting a warm pool on functions can lead to increased costs.

        ```python
        # Usage on a regular function.
        f = modal.Function.lookup("my-app", "function")
        f.keep_warm(2)

        # Usage on a parametrized function.
        Model = modal.Cls.lookup("my-app", "Model")
        Model("fine-tuned-model").keep_warm(2)
        ```
        """
        if self._is_method:
            raise InvalidError(
                textwrap.dedent(
                    """
                The `.keep_warm()` method can not be used on Modal class *methods* deployed using Modal >v0.63.

                Call `.keep_warm()` on the class *instance* instead.
            """
                )
            )
        assert self._client and self._client.stub
        request = api_pb2.FunctionUpdateSchedulingParamsRequest(
            function_id=self._object_id, warm_pool_size_override=warm_pool_size
        )
        await retry_transient_errors(self._client.stub.FunctionUpdateSchedulingParams, request)

    @classmethod
    def from_name(
        cls: Type["_Function"],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
    ) -> "_Function":
        """Retrieve a function with a given name and tag.

        ```python
        other_function = modal.Function.from_name("other-app", "function")
        ```
        """

        async def _load_remote(self: _Function, resolver: Resolver, existing_object_id: Optional[str]):
            assert resolver.client and resolver.client.stub
            request = api_pb2.FunctionGetRequest(
                app_name=app_name,
                object_tag=tag or "",
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver) or "",
            )
            try:
                response = await retry_transient_errors(resolver.client.stub.FunctionGet, request)
            except GRPCError as exc:
                if exc.status == Status.NOT_FOUND:
                    raise NotFoundError(exc.message)
                else:
                    raise

            self._hydrate(response.function_id, resolver.client, response.handle_metadata)

        rep = f"Ref({app_name})"
        return cls._from_loader(_load_remote, rep, is_another_app=True)

    @staticmethod
    async def lookup(
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ) -> "_Function":
        """Lookup a function with a given name and tag.

        ```python
        other_function = modal.Function.lookup("other-app", "function")
        ```
        """
        obj = _Function.from_name(app_name, tag, namespace=namespace, environment_name=environment_name)
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

    def get_build_def(self) -> str:
        """mdmd:hidden"""
        assert hasattr(self, "_raw_f") and hasattr(self, "_build_args")
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
        self._all_mounts = []  # used for file watching
        self._use_function_id = ""

    def _hydrate_metadata(self, metadata: Optional[Message]):
        # Overridden concrete implementation of base class method
        assert metadata and isinstance(metadata, (api_pb2.Function, api_pb2.FunctionHandleMetadata))
        self._is_generator = metadata.function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR
        self._web_url = metadata.web_url
        self._function_name = metadata.function_name
        self._is_method = metadata.is_method
        self._use_function_id = metadata.use_function_id
        self._use_method_name = metadata.use_method_name

    def _invocation_function_id(self) -> str:
        return self._use_function_id or self.object_id

    def _get_metadata(self):
        # Overridden concrete implementation of base class method
        assert self._function_name
        return api_pb2.FunctionHandleMetadata(
            function_name=self._function_name,
            function_type=(
                api_pb2.Function.FUNCTION_TYPE_GENERATOR
                if self._is_generator
                else api_pb2.Function.FUNCTION_TYPE_FUNCTION
            ),
            web_url=self._web_url or "",
            use_method_name=self._use_method_name,
            use_function_id=self._use_function_id,
            is_method=self._is_method,
        )

    def _set_mute_cancellation(self, value: bool = True):
        self._mute_cancellation = value

    def _set_output_mgr(self, output_mgr: OutputManager):
        self._output_mgr = output_mgr

    @property
    def web_url(self) -> str:
        """URL of a Function running as a web endpoint."""
        if not self._web_url:
            raise ValueError(
                f"No web_url can be found for function {self._function_name}. web_url "
                "can only be referenced from a running app context"
            )
        return self._web_url

    @property
    def is_generator(self) -> bool:
        """mdmd:hidden"""
        assert self._is_generator is not None
        return self._is_generator

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
        if self._web_url:
            raise InvalidError(
                "A web endpoint function cannot be directly invoked for parallel remote execution. "
                f"Invoke this function via its web url '{self._web_url}' or call it locally: {self._function_name}()."
            )
        if self._is_generator:
            raise InvalidError("A generator function cannot be called with `.map(...)`.")

        assert self._function_name
        count_update_callback = (
            self._output_mgr.function_progress_callback(self._function_name, total=None) if self._output_mgr else None
        )

        async for item in _map_invocation(
            self,  # type: ignore
            input_queue,
            self._client,
            order_outputs,
            return_exceptions,
            count_update_callback,
        ):
            yield item

    async def _call_function(self, args, kwargs):
        invocation = await _Invocation.create(self, args, kwargs, client=self._client)
        try:
            return await invocation.run_function()
        except asyncio.CancelledError:
            # this can happen if the user terminates a program, triggering a cancellation cascade
            if not self._mute_cancellation:
                raise

    async def _call_function_nowait(self, args, kwargs) -> _Invocation:
        return await _Invocation.create(self, args, kwargs, client=self._client)

    @warn_if_generator_is_not_consumed()
    @live_method_gen
    @synchronizer.no_input_translation
    async def _call_generator(self, args, kwargs):
        invocation = await _Invocation.create(self, args, kwargs, client=self._client)
        async for res in invocation.run_generator():
            yield res

    @synchronizer.no_io_translation
    async def _call_generator_nowait(self, args, kwargs):
        return await _Invocation.create(self, args, kwargs, client=self._client)

    @synchronizer.no_io_translation
    @live_method
    async def remote(self, *args, **kwargs) -> Any:
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

    @synchronizer.no_io_translation
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

    @synchronizer.no_io_translation
    @live_method
    async def shell(self, *args, **kwargs) -> None:
        if self._is_generator:
            async for item in self._call_generator(args, kwargs):
                pass
        else:
            await self._call_function(args, kwargs)

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
    def local(self, *args, **kwargs) -> Any:
        """
        Calls the function locally, executing it with the given arguments and returning the execution's result.

        The function will execute in the same environment as the caller, just like calling the underlying function
        directly in Python. In particular, only secrets available in the caller environment will be available
        through environment variables.
        """
        # TODO(erikbern): it would be nice to remove the nowrap thing, but right now that would cause
        # "user code" to run on the synchronicity thread, which seems bad
        info = self._get_info()

        if is_local() and self.spec.volumes or self.spec.network_file_systems:
            warnings.warn(
                f"The {info.function_name} function is executing locally "
                + "and will not have access to the mounted Volume or NetworkFileSystem data"
            )
        if not info or not info.raw_f:
            msg = (
                "The definition for this function is missing so it is not possible to invoke it locally. "
                "If this function was retrieved via `Function.lookup` you need to use `.remote()`."
            )
            raise ExecutionError(msg)

        obj: Optional["modal.cls._Obj"] = self._get_obj()

        if not obj:
            fun = info.raw_f
            return fun(*args, **kwargs)
        else:
            # This is a method on a class, so bind the self to the function
            user_cls_instance = obj._get_user_cls_instance()

            fun = info.raw_f.__get__(user_cls_instance)

            if is_async(info.raw_f):
                # We want to run __aenter__ and fun in the same coroutine
                async def coro():
                    await obj.aenter()
                    return await fun(*args, **kwargs)

                return coro()
            else:
                obj.enter()
                return fun(*args, **kwargs)

    @synchronizer.no_input_translation
    @live_method
    async def spawn(self, *args, **kwargs) -> Optional["_FunctionCall"]:
        """Calls the function with the given arguments, without waiting for the results.

        Returns a `modal.functions.FunctionCall` object, that can later be polled or
        waited for using `.get(timeout=...)`.
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
        return self._raw_f

    @live_method
    async def get_current_stats(self) -> FunctionStats:
        """Return a `FunctionStats` object describing the current function's queue and runner counts."""
        assert self._client.stub
        resp = await retry_transient_errors(
            self._client.stub.FunctionGetCurrentStats,
            api_pb2.FunctionGetCurrentStatsRequest(function_id=self.object_id),
            total_timeout=10.0,
        )
        return FunctionStats(backlog=resp.backlog, num_total_runners=resp.num_total_tasks)

    # A bit hacky - but the map-style functions need to not be synchronicity-wrapped
    # in order to not execute their input iterators on the synchronicity event loop.
    # We still need to wrap them using MethodWithAio to maintain a synchronicity-like
    # api with `.aio` and get working type-stubs and reference docs generation:
    map = MethodWithAio(_map_sync, _map_async, synchronizer)
    starmap = MethodWithAio(_starmap_sync, _starmap_async, synchronizer)
    for_each = MethodWithAio(_for_each_sync, _for_each_async, synchronizer)


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
        assert self._client.stub
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
        """Cancels the function call, which will stop its execution and mark its inputs as
        [`TERMINATED`](/docs/reference/modal.call_graph#modalcall_graphinputstatus)."""
        request = api_pb2.FunctionCallCancelRequest(function_call_id=self.object_id)
        assert self._client and self._client.stub
        await retry_transient_errors(self._client.stub.FunctionCallCancel, request)

    @staticmethod
    async def from_id(function_call_id: str, client: Optional[_Client] = None) -> "_FunctionCall":
        if client is None:
            client = await _Client.from_env()

        return _FunctionCall._new_hydrated(function_call_id, client, None)


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
        return await TaskContext.gather(*[fc.get() for fc in function_calls])
    except Exception as exc:
        # TODO: kill all running function calls
        raise exc


gather = synchronize_api(_gather)
