# Copyright Modal Labs 2026
import dataclasses
from collections.abc import Collection, Sequence, Sized
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from modal_proto import api_pb2

from ._resources import convert_fn_config_to_resources_config
from ._serialization import (
    apply_defaults,
    serialize,
    serialize_proto_params,
    validate_parameter_values,
)
from ._utils.function_utils import _parse_retries
from ._utils.mount_utils import validate_volumes, validate_volumes_by_object_id
from .cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from .retries import Retries
from .secret import _Secret
from .volume import _Volume, _volume_to_mount_proto

if TYPE_CHECKING:
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
        return [dep for dep in all_deps if not dep.is_hydrated]

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
            cloud_bucket_mounts=cloud_bucket_mounts_to_proto(self.cloud_bucket_mounts),
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

    assert base_function._client and base_function._client.stub
    response = await base_function._client.stub.FunctionBindParams(req)

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
        if not base_function.is_hydrated:
            await base_function.hydrate(load_context.client)
        assert base_function._client and base_function._client.stub

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
    fun._info = base_function._info
    fun._spec = base_function._spec  # TODO (elias): fix - this is incorrect when using with_options
    return fun
