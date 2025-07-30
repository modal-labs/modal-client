# Copyright Modal Labs 2025
import asyncio
import math
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Optional
from urllib.parse import urlparse

from modal.cls import _Cls
from modal.dict import _Dict
from modal_proto import api_pb2

from .._tunnel import _forward as _forward_tunnel
from .._utils.async_utils import synchronize_api, synchronizer
from .._utils.grpc_utils import retry_transient_errors
from ..client import _Client
from ..config import logger
from ..exception import InvalidError


class _FlashManager:
    def __init__(self, client: _Client, port: int, health_check_url: Optional[str] = None):
        self.client = client
        self.port = port
        self.health_check_url = health_check_url
        self.tunnel_manager = _forward_tunnel(port, client=client)
        self.stopped = False

    async def _start(self):
        self.tunnel = await self.tunnel_manager.__aenter__()

        parsed_url = urlparse(self.tunnel.url)
        host = parsed_url.hostname
        port = parsed_url.port or 443

        self.heartbeat_task = asyncio.create_task(self._run_heartbeat(host, port))

    async def _run_heartbeat(self, host: str, port: int):
        first_registration = True
        while True:
            try:
                resp = await self.client.stub.FlashContainerRegister(
                    api_pb2.FlashContainerRegisterRequest(
                        priority=10,
                        weight=5,
                        host=host,
                        port=port,
                    ),
                    timeout=10,
                )
                if first_registration:
                    logger.warning(f"[Modal Flash] Listening at {resp.url} over {self.tunnel.url}")
                    first_registration = False
            except asyncio.CancelledError:
                logger.warning("[Modal Flash] Shutting down...")
                break
            except Exception as e:
                logger.error(f"[Modal Flash] Heartbeat failed: {e}")

            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.warning("[Modal Flash] Shutting down...")
                break

    def get_container_url(self):
        # WARNING: Try not to use this method; we aren't sure if we will keep it.
        return self.tunnel.url

    async def stop(self):
        self.heartbeat_task.cancel()
        await retry_transient_errors(
            self.client.stub.FlashContainerDeregister,
            api_pb2.FlashContainerDeregisterRequest(),
        )

        self.stopped = True
        logger.warning(f"[Modal Flash] No longer accepting new requests on {self.tunnel.url}.")

        # NOTE(gongy): We skip calling TunnelStop to avoid interrupting in-flight requests.
        # It is up to the user to wait after calling .stop() to drain in-flight requests.

    async def close(self):
        if not self.stopped:
            await self.stop()

        logger.warning(f"[Modal Flash] Closing tunnel on {self.tunnel.url}.")
        await self.tunnel_manager.__aexit__(*sys.exc_info())


FlashManager = synchronize_api(_FlashManager)


@synchronizer.create_blocking
async def flash_forward(port: int, health_check_url: Optional[str] = None) -> _FlashManager:
    """
    Forward a port to the Modal Flash service, exposing that port as a stable web endpoint.

    This is a highly experimental method that can break or be removed at any time without warning.
    Do not use this method unless explicitly instructed to do so by Modal support.
    """
    client = await _Client.from_env()

    manager = _FlashManager(client, port, health_check_url)
    await manager._start()
    return manager


class _FlashPrometheusAutoscaler:
    _max_window_seconds = 60 * 60

    def __init__(
        self,
        client: _Client,
        app_name: str,
        cls_name: str,
        metrics_endpoint: str,
        target_metric: str,
        target_metric_value: float,
        min_containers: Optional[int],
        max_containers: Optional[int],
        scale_up_tolerance: float,
        scale_down_tolerance: float,
        scale_up_stabilization_window_seconds: int,
        scale_down_stabilization_window_seconds: int,
        autoscaling_interval_seconds: int,
    ):
        if scale_up_stabilization_window_seconds > self._max_window_seconds:
            raise InvalidError(
                f"scale_up_stabilization_window_seconds must be less than or equal to {self._max_window_seconds}"
            )
        if scale_down_stabilization_window_seconds > self._max_window_seconds:
            raise InvalidError(
                f"scale_down_stabilization_window_seconds must be less than or equal to {self._max_window_seconds}"
            )
        if target_metric_value <= 0:
            raise InvalidError("target_metric_value must be greater than 0")

        import aiohttp

        self.client = client
        self.app_name = app_name
        self.cls_name = cls_name
        self.metrics_endpoint = metrics_endpoint
        self.target_metric = target_metric
        self.target_metric_value = target_metric_value
        self.min_containers = min_containers
        self.max_containers = max_containers
        self.scale_up_tolerance = scale_up_tolerance
        self.scale_down_tolerance = scale_down_tolerance
        self.scale_up_stabilization_window_seconds = scale_up_stabilization_window_seconds
        self.scale_down_stabilization_window_seconds = scale_down_stabilization_window_seconds
        self.autoscaling_interval_seconds = autoscaling_interval_seconds

        FlashClass = _Cls.from_name(app_name, cls_name)
        self.fn = FlashClass._class_service_function
        self.cls = FlashClass()

        self.http_client = aiohttp.ClientSession()
        self.autoscaling_decisions_dict = _Dict.from_name(
            f"{app_name}-{cls_name}-autoscaling-decisions",
            create_if_missing=True,
        )

        self.autoscaler_thread = None

    async def start(self):
        await self.fn.hydrate(client=self.client)
        self.autoscaler_thread = asyncio.create_task(self._run_autoscaler_loop())

    async def _run_autoscaler_loop(self):
        while True:
            try:
                autoscaling_time = time.time()

                current_replicas = await self.autoscaling_decisions_dict.get("current_replicas", 0)
                autoscaling_decisions = await self.autoscaling_decisions_dict.get("autoscaling_decisions", [])
                if not isinstance(current_replicas, int):
                    logger.warning(f"[Modal Flash] Invalid item in autoscaling decisions: {current_replicas}")
                    current_replicas = 0
                if not isinstance(autoscaling_decisions, list):
                    logger.warning(f"[Modal Flash] Invalid item in autoscaling decisions: {autoscaling_decisions}")
                    autoscaling_decisions = []
                for item in autoscaling_decisions:
                    if (
                        not isinstance(item, tuple)
                        or len(item) != 2
                        or not isinstance(item[0], float)
                        or not isinstance(item[1], int)
                    ):
                        logger.warning(f"[Modal Flash] Invalid item in autoscaling decisions: {item}")
                        autoscaling_decisions = []
                        break

                autoscaling_decisions = [
                    (timestamp, decision)
                    for timestamp, decision in autoscaling_decisions
                    if timestamp >= autoscaling_time - self._max_window_seconds
                ]

                current_target_containers = await self._compute_target_containers(current_replicas)
                autoscaling_decisions.append((autoscaling_time, current_target_containers))

                actual_target_containers = self._make_scaling_decision(
                    current_replicas,
                    autoscaling_decisions,
                    scale_up_stabilization_window_seconds=self.scale_up_stabilization_window_seconds,
                    scale_down_stabilization_window_seconds=self.scale_down_stabilization_window_seconds,
                    min_containers=self.min_containers,
                    max_containers=self.max_containers,
                )

                logger.warning(
                    f"[Modal Flash] Scaling to {actual_target_containers} containers. Autoscaling decision "
                    f"made in {time.time() - autoscaling_time} seconds."
                )

                await self.autoscaling_decisions_dict.put(
                    "autoscaling_decisions",
                    autoscaling_decisions,
                )
                await self.autoscaling_decisions_dict.put("current_replicas", actual_target_containers)

                await self.cls.update_autoscaler(
                    min_containers=actual_target_containers,
                )

                if time.time() - autoscaling_time < self.autoscaling_interval_seconds:
                    await asyncio.sleep(self.autoscaling_interval_seconds - (time.time() - autoscaling_time))
            except asyncio.CancelledError:
                logger.warning("[Modal Flash] Shutting down autoscaler...")
                await self.http_client.close()
                break
            except Exception as e:
                logger.error(f"[Modal Flash] Error in autoscaler: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(self.autoscaling_interval_seconds)

    async def _compute_target_containers(self, current_replicas: int) -> int:
        containers = await self._get_all_containers()
        if len(containers) > current_replicas:
            logger.info(
                f"[Modal Flash] Current replicas {current_replicas} is less than the number of containers "
                f"{len(containers)}. Setting current_replicas = num_containers."
            )
            current_replicas = len(containers)

        if current_replicas == 0:
            return 1

        target_metric = self.target_metric
        target_metric_value = float(self.target_metric_value)

        sum_metric = 0
        containers_with_metrics = 0
        container_metrics_list = await asyncio.gather(
            *[
                self._get_metrics(f"https://{container.host}:{container.port}/{self.metrics_endpoint}")
                for container in containers
            ]
        )
        for container_metrics in container_metrics_list:
            if (
                container_metrics is None
                or target_metric not in container_metrics
                or len(container_metrics[target_metric]) == 0
            ):
                continue
            sum_metric += container_metrics[target_metric][0].value
            containers_with_metrics += 1

        n_containers_missing_metric = current_replicas - containers_with_metrics

        # Scale up / down conservatively: Any container that is missing the metric is assumed to be at the minimum
        # value of the metric when scaling up and the maximum value of the metric when scaling down.
        scale_up_target_metric_value = sum_metric / current_replicas
        scale_down_target_metric_value = (
            sum_metric + n_containers_missing_metric * target_metric_value
        ) / current_replicas

        scale_up_ratio = scale_up_target_metric_value / target_metric_value
        scale_down_ratio = scale_down_target_metric_value / target_metric_value

        desired_replicas = current_replicas
        if scale_up_ratio > 1 + self.scale_up_tolerance:
            desired_replicas = math.ceil(current_replicas * scale_up_ratio)
        elif scale_down_ratio < 1 - self.scale_down_tolerance:
            desired_replicas = math.ceil(current_replicas * scale_down_ratio)

        logger.warning(
            f"[Modal Flash] Current replicas: {current_replicas}, target metric value: {target_metric_value}, "
            f"current sum of metric values: {sum_metric}, number of containers missing metric: "
            f"{n_containers_missing_metric}, scale up ratio: {scale_up_ratio}, scale down ratio: {scale_down_ratio}, "
            f"desired replicas: {desired_replicas}"
        )

        return desired_replicas

    async def _get_metrics(self, url: str) -> Optional[dict[str, list[Any]]]:  # technically any should be Sample
        from prometheus_client.parser import Sample, text_string_to_metric_families

        # Fetch the metrics from the endpoint
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"[Modal Flash] Error getting metrics from {url}: {e}")
            return None

        # Parse the text-based Prometheus metrics format
        metrics: dict[str, list[Sample]] = defaultdict(list)
        for family in text_string_to_metric_families(await response.text()):
            for sample in family.samples:
                metrics[sample.name] += [sample]

        return metrics

    async def _get_all_containers(self):
        req = api_pb2.FlashContainerListRequest(function_id=self.fn.object_id)
        resp = await retry_transient_errors(self.client.stub.FlashContainerList, req)
        return resp.containers

    def _make_scaling_decision(
        self,
        current_replicas: int,
        autoscaling_decisions: list[tuple[float, int]],
        scale_up_stabilization_window_seconds: int = 0,
        scale_down_stabilization_window_seconds: int = 60 * 5,
        min_containers: Optional[int] = None,
        max_containers: Optional[int] = None,
    ) -> int:
        """
        Return the target number of containers following (simplified) Kubernetes HPA
        stabilization-window semantics.

        Args:
            current_replicas: Current number of running Pods/containers.
            autoscaling_decisions: List of (timestamp, desired_replicas) pairs, where
                                   timestamp is a UNIX epoch float (seconds).
                                   The list *must* contain at least one entry and should
                                   already include the most-recent measurement.
            scale_up_stabilization_window_seconds: 0 disables the up-window.
            scale_down_stabilization_window_seconds: 0 disables the down-window.
            min_containers / max_containers: Clamp the final decision to this range.

        Returns:
            The target number of containers.
        """
        if not autoscaling_decisions:
            # Without data we can’t make a new decision – stay where we are.
            return current_replicas

        # Sort just once in case the caller didn’t: newest record is last.
        autoscaling_decisions.sort(key=lambda rec: rec[0])
        now_ts, latest_desired = autoscaling_decisions[-1]

        if latest_desired > current_replicas:
            # ---- SCALE-UP path ----
            window_start = now_ts - scale_up_stabilization_window_seconds
            # Consider only records *inside* the window.
            desired_candidates = [desired for ts, desired in autoscaling_decisions if ts >= window_start]
            # Use the *minimum* so that any temporary dip blocks the scale-up.
            candidate = min(desired_candidates) if desired_candidates else latest_desired
            new_replicas = max(current_replicas, candidate)  # never scale *down* here
        elif latest_desired < current_replicas:
            # ---- SCALE-DOWN path ----
            window_start = now_ts - scale_down_stabilization_window_seconds
            desired_candidates = [desired for ts, desired in autoscaling_decisions if ts >= window_start]
            # Use the *maximum* so that any temporary spike blocks the scale-down.
            candidate = max(desired_candidates) if desired_candidates else latest_desired
            new_replicas = min(current_replicas, candidate)  # never scale *up* here
        else:
            # No change requested.
            new_replicas = current_replicas

        # Clamp to [min_containers, max_containers].
        if min_containers is not None:
            new_replicas = max(min_containers, new_replicas)
        if max_containers is not None:
            new_replicas = min(max_containers, new_replicas)
        return new_replicas

    async def stop(self):
        self.autoscaler_thread.cancel()
        await self.autoscaler_thread


FlashPrometheusAutoscaler = synchronize_api(_FlashPrometheusAutoscaler)


@synchronizer.create_blocking
async def flash_prometheus_autoscaler(
    app_name: str,
    cls_name: str,
    # Endpoint to fetch metrics from. Must be in Prometheus format. Example: "/metrics"
    metrics_endpoint: str,
    # Target metric to autoscale on. Example: "vllm:num_requests_running"
    target_metric: str,
    # Target metric value. Example: 25
    target_metric_value: float,
    min_containers: Optional[int] = None,
    max_containers: Optional[int] = None,
    # Corresponds to https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/#tolerance
    scale_up_tolerance: float = 0.1,
    # Corresponds to https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/#tolerance
    scale_down_tolerance: float = 0.1,
    # Corresponds to https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/#stabilization-window
    scale_up_stabilization_window_seconds: int = 0,
    # Corresponds to https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/#stabilization-window
    scale_down_stabilization_window_seconds: int = 300,
    # How often to make autoscaling decisions.
    # Corresponds to --horizontal-pod-autoscaler-sync-period in Kubernetes.
    autoscaling_interval_seconds: int = 15,
) -> _FlashPrometheusAutoscaler:
    """
    Autoscale a Flash service based on containers' Prometheus metrics.

    The package `prometheus_client` is required to use this method.

    This is a highly experimental method that can break or be removed at any time without warning.
    Do not use this method unless explicitly instructed to do so by Modal support.
    """

    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        raise ImportError("The package `prometheus_client` is required to use this method.")

    client = await _Client.from_env()
    autoscaler = _FlashPrometheusAutoscaler(
        client,
        app_name,
        cls_name,
        metrics_endpoint,
        target_metric,
        target_metric_value,
        min_containers,
        max_containers,
        scale_up_tolerance,
        scale_down_tolerance,
        scale_up_stabilization_window_seconds,
        scale_down_stabilization_window_seconds,
        autoscaling_interval_seconds,
    )
    await autoscaler.start()
    return autoscaler


@synchronizer.create_blocking
async def flash_get_containers(app_name: str, cls_name: str) -> list[dict[str, Any]]:
    """
    Return a list of flash containers for a deployed Flash service.

    This is a highly experimental method that can break or be removed at any time without warning.
    Do not use this method unless explicitly instructed to do so by Modal support.
    """
    client = await _Client.from_env()
    fn = _Cls.from_name(app_name, cls_name)._class_service_function
    assert fn is not None
    await fn.hydrate(client=client)
    req = api_pb2.FlashContainerListRequest(function_id=fn.object_id)
    resp = await retry_transient_errors(client.stub.FlashContainerList, req)
    return resp.containers
