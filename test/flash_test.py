# Copyright Modal Labs 2025
# pyright: reportMissingImports=false
import pytest
from types import MethodType
from typing import Any, Mapping, Optional, cast
from urllib.parse import urlparse

from modal.experimental.flash import _FlashPrometheusAutoscaler


class _DummyContainer:
    def __init__(self, host: str, port: int = 443):
        self.host = host
        self.port = port


class _DummySample:
    def __init__(self, value: float):
        self.value = value


def _make_autoscaler(
    metrics_by_host: Mapping[str, Optional[float]],
    container_hosts: list[str],
    *,
    target_metric_value: float = 10.0,
    scale_up_tolerance: float = 0.1,
    scale_down_tolerance: float = 0.1,
    target_metric: str = "test_metric",
    overprovision_containers: int | None = None,
):
    autoscaler = _FlashPrometheusAutoscaler.__new__(_FlashPrometheusAutoscaler)

    autoscaler_any = cast(Any, autoscaler)
    autoscaler_any.target_metric = target_metric
    autoscaler_any.target_metric_value = target_metric_value
    autoscaler_any.scale_up_tolerance = scale_up_tolerance
    autoscaler_any.scale_down_tolerance = scale_down_tolerance
    autoscaler_any.min_overprovision_containers = overprovision_containers
    autoscaler_any.metrics_endpoint = "metrics"

    containers = [_DummyContainer(h) for h in container_hosts]

    async def _get_all_containers(_self):
        return containers

    async def _get_metrics(_self, url: str):
        host = urlparse(url).hostname or ""
        value = metrics_by_host.get(host, None)
        if value is None:
            return None
        return {target_metric: [_DummySample(value)]}

    autoscaler._get_all_containers = MethodType(_get_all_containers, autoscaler)
    autoscaler._get_metrics = MethodType(_get_metrics, autoscaler)

    return autoscaler


@pytest.mark.asyncio
async def test_compute_target_containers_returns_one_when_current_zero():
    # No containers discovered; current_replicas == 0 should return 1
    autoscaler = _make_autoscaler(metrics_by_host={}, container_hosts=[])
    result = await autoscaler._compute_target_containers(current_replicas=0)
    assert result == 1


@pytest.mark.asyncio
async def test_compute_target_containers_scale_up_all_above_target():
    # 3 containers, all emit value 15 with target 10 -> avg=15 -> ratio=1.5 -> ceil(3*1.5)=5
    metrics = {"h1": 15.0, "h2": 15.0, "h3": 15.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
        container_hosts=["h1", "h2", "h3"],
        target_metric_value=10.0,
    )
    result = await autoscaler._compute_target_containers(current_replicas=3)
    assert result == 5


@pytest.mark.asyncio
async def test_compute_target_containers_scale_down_all_below_target():
    # 4 containers, all emit value 5 with target 10 -> avg=5 -> ratio=0.5 -> ceil(4*0.5)=2
    metrics = {"h1": 5.0, "h2": 5.0, "h3": 5.0, "h4": 5.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
        container_hosts=["h1", "h2", "h3", "h4"],
        target_metric_value=10.0,
    )
    result = await autoscaler._compute_target_containers(current_replicas=4)
    assert result == 2


@pytest.mark.asyncio
async def test_compute_target_containers_unhealthy_assumed_at_target():
    # 3 containers, only one emits 10 (target), two unhealthy -> both assumed at target for scale-up calc
    metrics = {"h1": 10.0, "h2": None, "h3": None}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
        container_hosts=["h1", "h2", "h3"],
        target_metric_value=10.0,
    )
    result = await autoscaler._compute_target_containers(current_replicas=3)
    # scale_up_ratio == scale_down_ratio == 1 -> no change
    assert result == 3


@pytest.mark.asyncio
async def test_compute_target_containers_current_less_than_discoverable():
    # current_replicas (1) < discoverable containers (3) -> adjusted up to 3; metrics equal target -> stay at 3
    metrics = {"h1": 10.0, "h2": 10.0, "h3": 10.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
        container_hosts=["h1", "h2", "h3"],
        target_metric_value=10.0,
    )
    result = await autoscaler._compute_target_containers(current_replicas=1)
    assert result == 3


@pytest.mark.asyncio
async def test_compute_target_containers_overprovision_reduces_scale_up():
    # Overprovision reduces denominator in scale-up avg: 3 containers at 15, overprov=1 ->
    # (45)/(3-1)=22.5 -> ratio=2.25 -> ceil(3*2.25)=7
    metrics = {"h1": 15.0, "h2": 15.0, "h3": 15.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
        container_hosts=["h1", "h2", "h3"],
        target_metric_value=10.0,
        overprovision_containers=1,
    )
    result = await autoscaler._compute_target_containers(current_replicas=3)
    assert result == 7


@pytest.mark.asyncio
async def test_overprovision_does_not_affect_scale_down():
    # All below target triggers scale-down; overprovision should not change scale-down computation
    metrics = {"h1": 5.0, "h2": 5.0, "h3": 5.0, "h4": 5.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
        container_hosts=["h1", "h2", "h3", "h4"],
        target_metric_value=10.0,
        overprovision_containers=2,
    )
    result = await autoscaler._compute_target_containers(current_replicas=4)
    assert result == 2


@pytest.mark.asyncio
async def test_overprovision_denominator_floored_to_one():
    # Overprovision greater than discoverable containers -> denominator would be <= 0 -> floored to 1
    metrics = {"h1": 15.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
        container_hosts=["h1"],
        target_metric_value=10.0,
        overprovision_containers=5,
    )
    result = await autoscaler._compute_target_containers(current_replicas=1)
    # ratio 15/10 = 1.5 -> ceil(1 * 1.5) = 2
    assert result == 2


@pytest.mark.asyncio
async def test_overprovision_interacts_with_unhealthy_scale_up():
    # One healthy at target, one unhealthy; with overprovision=1 denominator becomes 1, leading to scale-up.
    metrics = {"h1": 10.0, "h2": None}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
        container_hosts=["h1", "h2"],
        target_metric_value=10.0,
        overprovision_containers=1,
    )
    result = await autoscaler._compute_target_containers(current_replicas=2)
    # sum_metric=10, unhealthy=1 -> numerator=20, denominator=(1+1-1)=1 -> value=20 -> ratio=2 -> ceil(2*2)=4
    assert result == 4
