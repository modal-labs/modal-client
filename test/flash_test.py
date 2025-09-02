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
    metrics_by_host: Mapping[str, Optional[float]] = {},
    overprovision_containers: int = 0,
):
    autoscaler = _FlashPrometheusAutoscaler.__new__(_FlashPrometheusAutoscaler)

    autoscaler_any = cast(Any, autoscaler)
    autoscaler_any.target_metric = "test_metric"
    autoscaler_any.target_metric_value = 10.0
    autoscaler_any.scale_up_tolerance = 0.1
    autoscaler_any.scale_down_tolerance = 0.1
    autoscaler_any.min_overprovision_containers = overprovision_containers
    autoscaler_any.metrics_endpoint = "metrics"

    def _get_all_containers(_self):
        return [_DummyContainer(h) for h in metrics_by_host.keys()]

    def _get_metrics(_self, url: str):
        host = urlparse(url).hostname or ""
        value = metrics_by_host.get(host, None)
        if value is None:
            return None
        return {"test_metric": [_DummySample(value)]}

    autoscaler._get_all_containers = MethodType(_get_all_containers, autoscaler)  # type: ignore
    autoscaler._get_metrics = MethodType(_get_metrics, autoscaler)  # type: ignore

    return autoscaler


@pytest.mark.asyncio
async def test_compute_target_containers_returns_one_when_current_zero():
    # No containers discovered; current_replicas == 0 should return 1
    autoscaler = _make_autoscaler(metrics_by_host={})
    result = await autoscaler._compute_target_containers(current_replicas=0)
    assert result == 1


@pytest.mark.asyncio
async def test_compute_target_containers_scale_up_all_above_target():
    # 3 containers, all emit value 15 with target 10 -> avg=15 -> ratio=1.5 -> ceil(3*1.5)=5
    metrics = {"h1": 15.0, "h2": 15.0, "h3": 15.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
    )
    result = await autoscaler._compute_target_containers(current_replicas=3)
    assert result == 5


@pytest.mark.asyncio
async def test_compute_target_containers_scale_down_all_below_target():
    # 4 containers, all emit value 5 with target 10 -> avg=5 -> ratio=0.5 -> ceil(4*0.5)=2
    metrics = {"h1": 5.0, "h2": 5.0, "h3": 5.0, "h4": 5.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
    )
    result = await autoscaler._compute_target_containers(current_replicas=4)
    assert result == 2


@pytest.mark.asyncio
async def test_compute_target_containers_unhealthy_assumed_at_target():
    # 3 containers, only one emits 10 (target), two unhealthy -> both assumed at target for scale-up calc
    metrics = {"h1": 10.0, "h2": None, "h3": None}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
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
    )
    result = await autoscaler._compute_target_containers(current_replicas=1)
    assert result == 3


@pytest.mark.asyncio
async def test_compute_target_containers_overprovision_reduces_scale_up():
    # Overprovision reduces denominator in scale-up avg: 3 containers at 15, overprov=1 ->
    # (45)/(3-1)=22.5 -> ratio=2.25 -> ceil((3-1)*2.25)=5
    metrics = {"h1": 15.0, "h2": 15.0, "h3": 15.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
        overprovision_containers=1,
    )
    result = await autoscaler._compute_target_containers(current_replicas=3)
    assert result == 5


@pytest.mark.asyncio
async def test_overprovision_does_not_affect_scale_down():
    # All below target triggers scale-down; overprovision should not change scale-down computation
    metrics = {"h1": 5.0, "h2": 5.0, "h3": 5.0, "h4": 5.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
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
        overprovision_containers=1,
    )
    result = await autoscaler._compute_target_containers(current_replicas=2)
    # sum_metric=10, unhealthy=1 -> numerator=20, denominator=(1+1-1)=1 -> value=20 -> ratio=2 -> ceil(1*2)=2
    assert result == 3


@pytest.mark.asyncio
async def test_unhealthy_hosts_with_unhealthy_containers():
    """Example: Create 5 containers where h2 and h4 are unhealthy, others have metric value 15."""
    metrics = {"h1": 15.0, "h2": None, "h3": 15.0, "h4": None, "h5": 15.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
    )
    result = await autoscaler._compute_target_containers(current_replicas=5)
    # 3 healthy containers at 15.0, 2 unhealthy assumed at target (10.0)
    # avg = (15*3 + 10*2) / 5 = 65/5 = 13 -> ratio = 1.3 -> ceil(5*1.3) = 7
    assert result == 7


@pytest.mark.asyncio
async def test_unhealthy_hosts_with_mixed_metrics():
    """Example: Create containers with different metric values and some unhealthy."""
    metrics = {"h1": 12.0, "h2": 8.0, "h3": 15.0, "h4": None, "h5": None}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
    )
    result = await autoscaler._compute_target_containers(current_replicas=5)
    # 3 healthy: 12+8+15=35, 2 unhealthy assumed at 10 each: 20
    # avg = (35 + 20) / 5 = 11 -> ratio = 1.1 -> ceil(5*1.1) = 6
    assert result == 6


@pytest.mark.asyncio
async def test_unhealthy_hosts_simple_all_same_value():
    """Example: Create 4 containers all with the same metric value."""
    autoscaler = _make_autoscaler(
        metrics_by_host={"h1": 5.0, "h2": 5.0, "h3": 5.0, "h4": 5.0},
    )
    result = await autoscaler._compute_target_containers(current_replicas=4)
    # All containers at 5.0, below target -> scale down
    # avg = 5.0 -> ratio = 0.5 -> ceil(4*0.5) = 2
    assert result == 2


@pytest.mark.asyncio
async def test_unhealthy_hosts_with_custom_numbers():
    """Example: Custom target values, tolerances, and overprovision settings."""
    autoscaler = _make_autoscaler(
        metrics_by_host={"h1": 25.0, "h2": 30.0, "h3": None},
        overprovision_containers=1,
    )
    result = await autoscaler._compute_target_containers(current_replicas=3)
    # 2 healthy: 25+30=55, 1 unhealthy assumed at 1.1*target: 11
    # With overprovision=1: avg = (55 + 11) / (2) = 66/2 = 33
    # ratio = 33/10 = 3.3 -> ceil(2*3.3) = 7
    assert result == 7


@pytest.mark.asyncio
async def test_all_unhealthy_no_metrics():
    """
    Scenario: 20 current replicas, all containers missing metrics (unhealthy), 17 provisioned containers.
    Target metric value: 20.0, scale up ratio: 1.1, scale down ratio: 1.0, desired replicas: 17.
    """
    # Simulate 20 containers, all missing metrics (None), 17 provisioned containers
    metrics = {f"h{i}": None for i in range(1, 21)}
    autoscaler = _make_autoscaler(metrics_by_host=metrics)
    autoscaler_any = cast(Any, autoscaler)
    autoscaler_any.target_metric_value = 20.0

    # Simulate the case where only 17 containers are provisioned (e.g., due to unhealthy ones)
    # This may require patching _get_all_containers or passing a param if the implementation supports it.
    # For this test, we assume _get_all_containers returns 17 containers.
    def _get_all_containers(_self):
        return [_DummyContainer(f"h{i}") for i in range(1, 18)]

    autoscaler._get_all_containers = MethodType(_get_all_containers, autoscaler)  # type: ignore

    result = await autoscaler._compute_target_containers(current_replicas=20)
    # All containers are unhealthy, so desired replicas should be 17 (provisioned)
    assert result == 17

    #  this is wrong  ???
