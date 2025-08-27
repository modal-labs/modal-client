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
    metrics_by_host: Optional[Mapping[str, Optional[float]]] = None,
    container_hosts: list[str] | None = None,
    *,
    # New flexible parameters
    num_containers: int | None = None,
    unhealthy_hosts: list[str] | None = None,
    metric_values: list[float] | float | None = None,
    default_metric_value: float = 10.0,
    # Existing parameters
    target_metric_value: float = 10.0,
    scale_up_tolerance: float = 0.1,
    scale_down_tolerance: float = 0.1,
    target_metric: str = "test_metric",
    overprovision_containers: int | None = None,
):
    # Build container_hosts and metrics_by_host from flexible parameters
    if metrics_by_host is None and container_hosts is None:
        # Use flexible parameters to build the configuration
        if num_containers is not None:
            container_hosts = [f"h{i + 1}" for i in range(num_containers)]
        else:
            container_hosts = []

        # Build metrics_by_host dictionary
        metrics_by_host = {}
        unhealthy_set = set(unhealthy_hosts or [])

        # Handle metric_values
        if isinstance(metric_values, (int, float)):
            # Single value for all healthy containers
            healthy_values = [metric_values] * len(container_hosts)
        elif isinstance(metric_values, list):
            # List of values for healthy containers
            healthy_values = metric_values[:]
        else:
            # Use default value for all healthy containers
            healthy_values = [default_metric_value] * len(container_hosts)

        # Assign metric values to containers
        healthy_index = 0
        for host in container_hosts:
            if host in unhealthy_set:
                metrics_by_host[host] = None  # Unhealthy container
            else:
                if healthy_index < len(healthy_values):
                    metrics_by_host[host] = healthy_values[healthy_index]
                    healthy_index += 1
                else:
                    metrics_by_host[host] = default_metric_value

    elif metrics_by_host is None or container_hosts is None:
        raise ValueError("Either provide both metrics_by_host and container_hosts, or use the flexible parameters")

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


# Helper functions demonstrating the new flexible API
def make_autoscaler_with_unhealthy(
    num_containers: int, unhealthy_hosts: list[str], metric_value: float = 10.0, **kwargs
):
    """Create autoscaler with specified number of containers and explicit unhealthy hosts."""
    return _make_autoscaler(
        num_containers=num_containers, unhealthy_hosts=unhealthy_hosts, metric_values=metric_value, **kwargs
    )


def make_autoscaler_with_mixed_metrics(healthy_values: list[float], unhealthy_hosts: list[str] | None = None, **kwargs):
    """Create autoscaler with different metric values for each healthy container."""
    num_containers = len(healthy_values) + len(unhealthy_hosts or [])
    return _make_autoscaler(
        num_containers=num_containers, unhealthy_hosts=unhealthy_hosts, metric_values=healthy_values, **kwargs
    )


def make_autoscaler_simple(num_containers: int, all_metric_value: float, **kwargs):
    """Create autoscaler with all containers having the same metric value."""
    return _make_autoscaler(num_containers=num_containers, metric_values=all_metric_value, **kwargs)


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
    # (45)/(3-1)=22.5 -> ratio=2.25 -> ceil((3-1)*2.25)=5
    metrics = {"h1": 15.0, "h2": 15.0, "h3": 15.0}
    autoscaler = _make_autoscaler(
        metrics_by_host=metrics,
        container_hosts=["h1", "h2", "h3"],
        target_metric_value=10.0,
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
    # sum_metric=10, unhealthy=1 -> numerator=20, denominator=(1+1-1)=1 -> value=20 -> ratio=2 -> ceil(1*2)=2
    assert result == 3


@pytest.mark.asyncio
async def test_unhealthy_hosts_with_unhealthy_containers():
    """Example: Create 5 containers where h2 and h4 are unhealthy, others have metric value 15."""
    autoscaler = make_autoscaler_with_unhealthy(
        num_containers=5,
        unhealthy_hosts=["h2", "h4"],
        metric_value=15.0,
        target_metric_value=10.0,
    )
    result = await autoscaler._compute_target_containers(current_replicas=5)
    # 3 healthy containers at 15.0, 2 unhealthy assumed at target (10.0)
    # avg = (15*3 + 10*2) / 5 = 65/5 = 13 -> ratio = 1.3 -> ceil(5*1.3) = 7
    assert result == 7


@pytest.mark.asyncio
async def test_unhealthy_hosts_with_mixed_metrics():
    """Example: Create containers with different metric values and some unhealthy."""
    autoscaler = make_autoscaler_with_mixed_metrics(
        healthy_values=[12.0, 8.0, 15.0],  # Different values for healthy containers
        unhealthy_hosts=["h4", "h5"],  # Two unhealthy containers
        target_metric_value=10.0,
    )
    result = await autoscaler._compute_target_containers(current_replicas=5)
    # 3 healthy: 12+8+15=35, 2 unhealthy assumed at 10 each: 20
    # avg = (35 + 20) / 5 = 11 -> ratio = 1.1 -> ceil(5*1.1) = 6
    assert result == 6


@pytest.mark.asyncio
async def test_unhealthy_hosts_simple_all_same_value():
    """Example: Create 4 containers all with the same metric value."""
    autoscaler = make_autoscaler_simple(
        num_containers=4,
        all_metric_value=5.0,
        target_metric_value=10.0,
    )
    result = await autoscaler._compute_target_containers(current_replicas=4)
    # All containers at 5.0, below target -> scale down
    # avg = 5.0 -> ratio = 0.5 -> ceil(4*0.5) = 2
    assert result == 2


@pytest.mark.asyncio
async def test_unhealthy_hosts_with_custom_numbers():
    """Example: Custom target values, tolerances, and overprovision settings."""
    autoscaler = _make_autoscaler(
        num_containers=3,
        unhealthy_hosts=["h1"],
        metric_values=[25.0, 30.0],  # Two healthy containers with high values
        target_metric_value=20.0,
        scale_up_tolerance=0.2,
        scale_down_tolerance=0.15,
        overprovision_containers=1,
    )
    result = await autoscaler._compute_target_containers(current_replicas=3)
    # 2 healthy: 25+30=55, 1 unhealthy assumed at target: 20
    # With overprovision=1: avg = (55 + 20) / (2+1-1) = 75/2 = 37.5
    # ratio = 37.5/20 = 1.875 -> ceil(2*1.875) = 4
    assert result == 4
