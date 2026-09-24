# Copyright Modal Labs 2026
import copy
import dataclasses
from collections.abc import Collection, Sequence, Sized
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, TypedDict

import google.protobuf.message

from modal_proto import api_pb2

from ._resources import convert_fn_config_to_resources_config
from ._serialization import (
    apply_defaults,
    deserialize_proto_params,
    serialize,
    serialize_proto_params,
    validate_parameter_values,
)
from ._utils.function_utils import _parse_retries
from ._utils.mount_utils import validate_volumes, validate_volumes_by_object_id
from .cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from .exception import InvalidError
from .retries import Retries
from .secret import _Secret
from .types import CloudBucketMountInfo, FunctionInfo, VolumeMountInfo
from .volume import _Volume, _volume_to_mount_proto

if TYPE_CHECKING:
    from modal.client import _Client

    from ._functions import _Function
    from ._load_context import LoadContext
    from ._object import _Object
    from ._resolver import Resolver


@dataclasses.dataclass()
class _FunctionOptions:
    """Data class that holds local state for a dynamically configured Function / Cls.

    Not a public interface. Dataclass fields represent post-validation parameter values.
    Use the `.new()` constructor to transform from the public interface types.
    """

    # Note that default values must be "untruthy" so we can that detect when they are not set.
    secrets: Collection[_Secret] = ()
    validated_volumes: Sequence[tuple[str, _Volume]] = ()
    cloud_bucket_mounts: Sequence[tuple[str, _CloudBucketMount]] = ()
    resources: api_pb2.Resources | None = None
    retry_policy: api_pb2.FunctionRetryPolicy | None = None
    max_containers: int | None = None
    buffer_containers: int | None = None
    scaledown_window: int | None = None
    timeout_secs: int | None = None
    scheduler_placement: api_pb2.SchedulerPlacement | None = None
    cloud: str | None = None
    max_concurrent_inputs: int | None = None
    target_concurrent_inputs: int | None = None
    batch_max_size: int | None = None
    batch_wait_ms: int | None = None
    routing_region: str | None = None

    @classmethod
    def new(
        cls,
        *,
        cpu: float | tuple[float, float] | None = None,
        memory: int | tuple[int, int] | None = None,
        gpu: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        volumes: dict[str | PurePosixPath, _Volume | _CloudBucketMount] = {},
        retries: int | Retries | None = None,
        max_containers: int | None = None,
        buffer_containers: int | None = None,
        scaledown_window: int | None = None,
        timeout: int | None = None,
        region: str | Sequence[str] | None = None,
        cloud: str | None = None,
        max_concurrent_inputs: int | None = None,
        target_concurrent_inputs: int | None = None,
        batch_max_size: int | None = None,
        batch_wait_ms: int | None = None,
        routing_region: str | None = None,
    ) -> "_FunctionOptions":
        """Internal constructor that validates and normalizes public parameters."""
        retry_policy = _parse_retries(retries)
        if gpu or cpu or memory:
            resources = convert_fn_config_to_resources_config(cpu=cpu, memory=memory, gpu=gpu)
        else:
            resources = None

        validated_volumes = validate_volumes(volumes)
        cloud_bucket_mounts = [(k, v) for k, v in validated_volumes if isinstance(v, _CloudBucketMount)]
        validated_volumes_no_cloud_buckets = [(k, v) for k, v in validated_volumes if isinstance(v, _Volume)]

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        scheduler_placement: api_pb2.SchedulerPlacement | None = None
        if region:
            regions = [region] if isinstance(region, str) else list(region)
            scheduler_placement = api_pb2.SchedulerPlacement(regions=regions)

        # Use batched and concurrent decorators to apply consistent validation logic
        from .partial_function import batched, concurrent

        if batch_max_size is not None and batch_wait_ms is not None:
            batched(max_batch_size=batch_max_size, wait_ms=batch_wait_ms)

        if max_concurrent_inputs:
            concurrent(max_inputs=max_concurrent_inputs, target_inputs=target_concurrent_inputs)

        return cls(
            secrets=secrets,
            validated_volumes=validated_volumes_no_cloud_buckets,
            cloud_bucket_mounts=cloud_bucket_mounts,
            resources=resources,
            retry_policy=retry_policy,
            max_containers=max_containers,
            buffer_containers=buffer_containers,
            scaledown_window=scaledown_window,
            timeout_secs=timeout,
            scheduler_placement=scheduler_placement,
            cloud=cloud,
            max_concurrent_inputs=max_concurrent_inputs,
            target_concurrent_inputs=target_concurrent_inputs,
            batch_max_size=batch_max_size,
            batch_wait_ms=batch_wait_ms,
            routing_region=routing_region,
        )

    def merge_options(self, new_options: "_FunctionOptions") -> "_FunctionOptions":
        """Implement protobuf-like MergeFrom semantics for this dataclass.

        This mostly exists to support "stacking" of `.with_options()` calls.
        Returns a new _FunctionOptions instance without modifying self.
        """
        # Create a shallow copy of self to start with.
        merged = dataclasses.replace(self)

        # Don't use dataclasses.asdict() because it does a deepcopy(), which chokes on a hydrated object.
        new_options_dict = {k.name: getattr(new_options, k.name) for k in dataclasses.fields(new_options)}

        # Resources needs special merge handling because individual fields are parameters in the public API.
        merged_resources = api_pb2.Resources()
        if merged.resources:
            merged_resources.MergeFrom(merged.resources)
        if new_resources := new_options_dict.pop("resources"):
            merged_resources.MergeFrom(new_resources)
        merged.resources = merged_resources

        for key, value in new_options_dict.items():
            if value:  # Only overwrite data when the value was set in the new options.
                setattr(merged, key, value)

        return merged

    def _unhydrated_object_deps(self) -> list["_Object"]:
        """Return unhydrated `modal.Object` instances that are part of the configuration payload."""
        all_deps = (
            [volume for _, volume in self.validated_volumes]
            + list(self.secrets)
            + [mount.secret for _, mount in self.cloud_bucket_mounts if mount.secret]
        )
        return [dep for dep in all_deps if not dep._is_hydrated]

    def to_proto(self) -> api_pb2.FunctionOptions:
        """Convert the dataclass to a FunctionOptions protobuf message."""
        # Validate that the same volume (by object_id) isn't mounted at multiple paths.
        # Needs to be called late so that volumes are hydrated
        validate_volumes_by_object_id(self.validated_volumes)

        volume_mounts = [_volume_to_mount_proto(path, volume) for path, volume in self.validated_volumes]
        return api_pb2.FunctionOptions(
            secret_ids=[secret.object_id for secret in self.secrets],
            replace_secret_ids=bool(self.secrets),
            replace_volume_mounts=len(volume_mounts) > 0,
            volume_mounts=volume_mounts,
            cloud_bucket_mounts=cloud_bucket_mounts_to_proto(self.cloud_bucket_mounts)[0],
            replace_cloud_bucket_mounts=bool(self.cloud_bucket_mounts),
            resources=self.resources,
            retry_policy=self.retry_policy,
            concurrency_limit=self.max_containers,
            buffer_containers=self.buffer_containers,
            task_idle_timeout_secs=self.scaledown_window,
            timeout_secs=self.timeout_secs,
            max_concurrent_inputs=self.max_concurrent_inputs,
            target_concurrent_inputs=self.target_concurrent_inputs,
            batch_max_size=self.batch_max_size,
            batch_linger_ms=self.batch_wait_ms,
            scheduler_placement=self.scheduler_placement,
            cloud_provider_str=self.cloud,
            routing_region=self.routing_region,
        )


@dataclasses.dataclass(frozen=True)
class _FunctionOptionsInfo:
    """Configuration overrides applied to a Function variant.

    Fields are named after the parameters of the builder methods that set them: `.with_options()`,
    `.with_concurrency()` (`max_inputs`, `target_inputs`), and `.with_batching()` (`max_batch_size`,
    `wait_ms`). Fields are None when the variant does not override that setting.
    """

    cpu: float | tuple[float, float] | None = None
    memory: int | tuple[int, int] | None = None
    gpu: str | None = None
    secrets: list[str] | None = None
    volumes: dict[str, VolumeMountInfo] | None = None  # mount path -> volume and its mount options
    cloud_bucket_mounts: dict[str, CloudBucketMountInfo] | None = None  # mount path -> bucket mount
    retries: Retries | None = None
    max_containers: int | None = None
    buffer_containers: int | None = None
    scaledown_window: int | None = None
    timeout: int | None = None
    region: str | list[str] | None = None
    cloud: str | None = None
    routing_region: str | None = None
    max_inputs: int | None = None
    target_inputs: int | None = None
    max_batch_size: int | None = None
    wait_ms: int | None = None

    @classmethod
    def _from_proto(cls, proto: api_pb2.FunctionOptions) -> "_FunctionOptionsInfo":
        cpu: float | tuple[float, float] | None = None
        memory: int | tuple[int, int] | None = None
        gpu: str | None = None
        if proto.HasField("resources"):
            resources = proto.resources
            if resources.milli_cpu_max > 0:
                cpu = (resources.milli_cpu / 1000, resources.milli_cpu_max / 1000)
            elif resources.milli_cpu > 0:
                cpu = resources.milli_cpu / 1000

            if resources.memory_mb_max > 0:
                memory = (resources.memory_mb, resources.memory_mb_max)
            elif resources.memory_mb > 0:
                memory = resources.memory_mb

            if resources.gpu_config.count > 0:
                gpu_config = resources.gpu_config
                gpu = gpu_config.gpu_type if gpu_config.count == 1 else f"{gpu_config.gpu_type}:{gpu_config.count}"

        retries: Retries | None = None
        if proto.HasField("retry_policy"):
            retry_policy = proto.retry_policy
            retries = Retries(
                max_retries=retry_policy.retries,
                backoff_coefficient=retry_policy.backoff_coefficient or 2.0,
                initial_delay=retry_policy.initial_delay_ms / 1000,
                max_delay=retry_policy.max_delay_ms / 1000 or 60.0,
            )

        region: str | list[str] | None = None
        if proto.HasField("scheduler_placement") and proto.scheduler_placement.regions:
            region = list(proto.scheduler_placement.regions)

        return cls(
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            secrets=list(proto.secret_ids) or None,
            volumes={
                vm.mount_path: VolumeMountInfo(
                    name=None,  # The mount records which Volume, not what it was called.
                    volume_id=vm.volume_id,
                    read_only=vm.read_only,
                    sub_path=vm.sub_path if vm.HasField("sub_path") else None,
                )
                for vm in proto.volume_mounts
            }
            or None,
            cloud_bucket_mounts={
                cbm.mount_path: CloudBucketMountInfo._from_proto(cbm) for cbm in proto.cloud_bucket_mounts
            }
            or None,
            retries=retries,
            max_containers=proto.concurrency_limit if proto.HasField("concurrency_limit") else None,
            buffer_containers=proto.buffer_containers if proto.HasField("buffer_containers") else None,
            scaledown_window=proto.task_idle_timeout_secs if proto.HasField("task_idle_timeout_secs") else None,
            timeout=proto.timeout_secs if proto.HasField("timeout_secs") else None,
            region=region,
            cloud=proto.cloud_provider_str if proto.HasField("cloud_provider_str") else None,
            routing_region=proto.routing_region if proto.HasField("routing_region") else None,
            max_inputs=proto.max_concurrent_inputs if proto.HasField("max_concurrent_inputs") else None,
            target_inputs=proto.target_concurrent_inputs if proto.HasField("target_concurrent_inputs") else None,
            max_batch_size=proto.batch_max_size if proto.HasField("batch_max_size") else None,
            wait_ms=proto.batch_linger_ms if proto.HasField("batch_linger_ms") else None,
        )


@dataclasses.dataclass(frozen=True)
class _FunctionVariantInfo:
    """Information about a variant of a Function, created by parametrization or `.with_options()`."""

    function_id: str
    parameters: dict[str, str | int | bytes | bool] | None  # None when the parameters can't be decoded
    options: _FunctionOptionsInfo | None

    @classmethod
    def _from_proto(cls, proto: api_pb2.FunctionVariantInfo) -> "_FunctionVariantInfo":
        parameters: dict[str, str | int | bytes | bool] | None
        try:
            parameters = deserialize_proto_params(proto.serialized_params)
        except (google.protobuf.message.DecodeError, InvalidError):
            parameters = None

        return cls(
            function_id=proto.function_id,
            parameters=parameters,
            options=_FunctionOptionsInfo._from_proto(proto.function_options)
            if proto.HasField("function_options")
            else None,
        )


@dataclasses.dataclass(frozen=True)
class _FunctionVariantListing:
    """The variants of a Function, and how the server chose to order them."""

    variants: list[_FunctionVariantInfo]
    # True when the variants are ordered by how many tasks each is running, busiest first.
    ordered_by_task_count: bool


async def _list_function_variants(
    client: "_Client", function_id: str, *, limit: int | None = None
) -> _FunctionVariantListing:
    variants: list[_FunctionVariantInfo] = []
    ordered_by_task_count = False
    cursor = None
    while True:
        request = api_pb2.FunctionListVariantsRequest(function_id=function_id, cursor=cursor, limit=limit or 0)
        response = await client._stub.FunctionListVariants(request)
        # An ordered listing arrives as a single response, so there is nothing to reconcile across
        # pages: every response of a multi-page listing reports itself unordered.
        ordered_by_task_count = response.ordered_by_task_count
        variants.extend(_FunctionVariantInfo._from_proto(info) for info in response.infos)
        if limit is not None and len(variants) >= limit:
            return _FunctionVariantListing(variants[:limit], ordered_by_task_count)
        if not response.HasField("next_cursor"):
            return _FunctionVariantListing(variants, ordered_by_task_count)
        cursor = response.next_cursor


async def _function_bind_params_cached(
    base_function: "_Function",
    req: api_pb2.FunctionBindParamsRequest,
) -> api_pb2.FunctionBindParamsResponse:
    """Cache layer for FunctionBindParams RPCs, scoped to a base Function handle.

    We have this because users probably do not realize that Function invocations structured as

        res = f.with_options(...).remote(...)

    would always need to do two sequential RPCs (bind params / call function variant).

    The bound Function ID from FunctionBindParams is deterministic with respect to the full request,
    so we can avoid the unnecessary call and hydrate the new instance from a cached response.

    The cache is stored on the base Function handle so that the variant cache behaves similarly to
    the local reference to the base Function metadata.

    """
    cache = base_function._bind_params_cache
    cache_key = req.SerializeToString(deterministic=True)

    cached_response = cache.get(cache_key)
    if cached_response is not None:
        cache.move_to_end(cache_key)
        response = api_pb2.FunctionBindParamsResponse()
        response.ParseFromString(cached_response)
        return response

    assert base_function._client and base_function._client._stub
    response = await base_function._client._stub.FunctionBindParams(req)

    cache[cache_key] = response.SerializeToString(deterministic=True)
    cache.move_to_end(cache_key)

    max_cache_size = 32
    while len(cache) > max_cache_size:
        cache.popitem(last=False)

    return response


def _make_function_variant(
    base_function: "_Function",
    options: _FunctionOptions | None,
    parameter_schema: Sequence[api_pb2.ClassParameterSpec] | None,
    args: Sized,
    kwargs: dict[str, Any],
) -> "_Function":
    """Extend a base Function with parameter values or dynamic configuration options."""

    async def _load(
        function_variant: "_Function",
        resolver: "Resolver",
        load_context: "LoadContext",
        existing_object_id: str | None,
    ):
        if not base_function._is_hydrated:
            await base_function.hydrate(load_context.client)

        assert base_function._client and base_function._client._stub

        if parameter_schema is None:
            # This branch is about backwards compatibility.
            # For Cls, we have `parameter_schema = None` for both old-style classes that
            # use a custom constructor (and hence use pickle serialization) and for
            # un-parameterized classes of any vintage, because such classes historically
            # sent serialized empty args/kwargs rather than a null `serialized_params` bytestring.
            serialized_params = serialize((args, kwargs))
        else:
            # New-style modal.parameter() based parameterization with protobuf serialization,
            # including true Function variants with no parameters defined
            # (in which case, serialized_params is a null bytestring).
            kwargs_with_defaults = apply_defaults(kwargs, parameter_schema)
            validate_parameter_values(kwargs_with_defaults, parameter_schema)
            serialized_params = serialize_proto_params(kwargs_with_defaults)

        options_pb = options.to_proto() if options else None

        req = api_pb2.FunctionBindParamsRequest(
            function_id=base_function.object_id,
            serialized_params=serialized_params,
            function_options=options_pb,
            environment_name=load_context.environment_name
            or "",  # TODO: investigate shouldn't environment name always be specified here?
        )

        response = await _function_bind_params_cached(base_function, req)
        function_variant._hydrate(response.bound_function_id, base_function._client, response.handle_metadata)

        if base_function._function_info is not None:
            if options is not None:
                function_variant._function_info = _get_function_info_with_options(base_function._function_info, options)
            else:
                function_variant._function_info = copy.deepcopy(base_function._function_info)

    def _deps():
        if options:
            return options._unhydrated_object_deps()
        return []

    fun = base_function._from_loader(
        _load,
        base_function._rep,
        hydrate_lazily=True,
        deps=_deps,
        load_context_overrides=base_function._load_context_overrides,
    )
    fun._source_info = base_function._source_info
    fun._spec = base_function._spec  # TODO (elias): fix - this is incorrect when using with_options

    # If we already have a ._function_info for the base function, we can pass it down - otherwise this
    # will be filled in during hydration
    if base_function._function_info is not None:
        if options is not None:
            fun._function_info = _get_function_info_with_options(base_function._function_info, options)
        else:
            fun._function_info = copy.deepcopy(base_function._function_info)

    return fun


class _FunctionInfoArgsT(TypedDict, total=False):
    cpu: float | tuple[float, float]
    memory_mib: int | tuple[int, int]
    gpus: list[tuple[str, int]]
    ephemeral_disk_mib: int
    timeout: int
    max_retries: int
    regions: list[str]
    cloud: str
    volumes: dict[str, VolumeMountInfo]
    cloud_bucket_mounts: dict[str, CloudBucketMountInfo]
    secrets: list[str]
    routing_region: str
    batching_info: FunctionInfo.BatchingInfo
    concurrency_info: FunctionInfo.ConcurrencyInfo


def _get_function_info_with_options(info: FunctionInfo, options: _FunctionOptions):
    args: _FunctionInfoArgsT = {}

    if options.resources is not None:
        res = options.resources
        if res.milli_cpu_max > 0:
            args["cpu"] = (res.milli_cpu / 1000, res.milli_cpu_max / 1000)
        elif res.milli_cpu > 0:
            args["cpu"] = res.milli_cpu / 1000

        if res.memory_mb_max > 0:
            args["memory_mib"] = (res.memory_mb, res.memory_mb_max)
        elif res.memory_mb > 0:
            args["memory_mib"] = res.memory_mb

        if res.HasField("gpu_config") and res.gpu_config.count > 0:
            args["gpus"] = [(res.gpu_config.gpu_type, res.gpu_config.count)]

        if res.ephemeral_disk_mb > 0:
            args["ephemeral_disk_mib"] = res.ephemeral_disk_mb

    if options.timeout_secs is not None:
        args["timeout"] = options.timeout_secs

    if options.retry_policy is not None:
        args["max_retries"] = options.retry_policy.retries

    if options.scheduler_placement is not None:
        sp = options.scheduler_placement
        # Note: not putting `nonpreemptible` here as that is not currently possible to set via
        # `.with_options(...)`
        if len(sp.regions) > 0:
            args["regions"] = list(sp.regions)

    if options.cloud is not None:
        args["cloud"] = options.cloud

    if len(options.validated_volumes) > 0:
        args["volumes"] = {
            mount_path: VolumeMountInfo(vol._name, vol._object_id, False, None)
            if vol._mount_options is None
            else VolumeMountInfo(
                vol._name,
                vol._object_id,
                vol._mount_options.read_only,
                vol._mount_options.sub_path,
            )
            for mount_path, vol in options.validated_volumes
        }

    if len(options.cloud_bucket_mounts) > 0:
        args["cloud_bucket_mounts"] = {}
        protos, _ = cloud_bucket_mounts_to_proto(options.cloud_bucket_mounts, include_secrets=False)

        for proto in protos:
            args["cloud_bucket_mounts"][proto.mount_path] = CloudBucketMountInfo._from_proto(proto)

    if len(options.secrets) > 0:
        args["secrets"] = [repr(s) for s in options.secrets]

    if options.routing_region:
        args["routing_region"] = options.routing_region

    if options.batch_max_size is not None and options.batch_wait_ms is not None:
        args["batching_info"] = FunctionInfo.BatchingInfo(
            max_batch_size=options.batch_max_size,
            wait_ms=options.batch_wait_ms,
        )

    if options.max_concurrent_inputs is not None or options.target_concurrent_inputs is not None:
        max_inputs = (
            options.max_concurrent_inputs
            if options.max_concurrent_inputs is not None
            else info.concurrency_info.max_inputs
            if info.concurrency_info is not None
            else None
        )

        target_inputs = (
            options.target_concurrent_inputs
            if options.target_concurrent_inputs is not None
            else info.concurrency_info.target_inputs
            if info.concurrency_info is not None
            else None
        )

        args["concurrency_info"] = FunctionInfo.ConcurrencyInfo(
            max_inputs=max_inputs,
            target_inputs=target_inputs,
        )

    return dataclasses.replace(copy.deepcopy(info), **args)
