# Copyright Modal Labs 2025
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from modal.experimental.flash import _FlashPrometheusAutoscaler


class TestFlashAutoscalerLogic:
    @pytest.fixture
    def autoscaler(self, client):
        """Single autoscaler instance shared across all tests."""
        with patch("aiohttp.ClientSession"):
            return _FlashPrometheusAutoscaler(
                client=client,
                app_name="test_app",
                cls_name="test_cls",
                metrics_endpoint="internal",
                target_metric="cpu_usage_percent",
                target_metric_value=0.5,
                min_containers=None,
                max_containers=None,
                scale_up_tolerance=0.1,
                scale_down_tolerance=0.1,
                scale_up_stabilization_window_seconds=0,
                scale_down_stabilization_window_seconds=300,
                autoscaling_interval_seconds=15,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flash_metric,target_value,metrics,current_replicas,expected_replicas",
        [
            ("cpu_usage_percent", 0.5, [0.8], 1, 2),  # High CPU single container
            ("cpu_usage_percent", 0.5, [0.2], 3, 2),  # Low CPU single container
            ("cpu_usage_percent", 0.5, [0.6, 0.7, 0.8, 0.9], 5, 8),  # High CPU multiple containers
            ("memory_usage_percent", 0.6, [0.9], 2, 3),  # High memory usage
            ("memory_usage_percent", 0.6, [0.52], 2, 2),  # Low memory usage within tolerance
        ],
    )
    async def test_metric_scaling(
        self, autoscaler, flash_metric, target_value, metrics, current_replicas, expected_replicas
    ):
        autoscaler.target_metric = flash_metric
        autoscaler.target_metric_value = target_value

        mock_containers = [MagicMock() for _ in range(len(metrics))]
        autoscaler._get_all_containers = AsyncMock(return_value=mock_containers)

        autoscaler._get_container_metrics = AsyncMock(
            side_effect=[MagicMock(metrics=MagicMock(**{flash_metric: value})) for value in metrics]
        )

        result = await autoscaler._compute_target_containers_internal(current_replicas=current_replicas)
        assert result == expected_replicas

    @pytest.mark.asyncio
    async def test_no_metrics_returns_current(self, autoscaler):
        mock_container = MagicMock()
        mock_container.id = "container_1"

        autoscaler._get_all_containers = AsyncMock(return_value=[mock_container])
        autoscaler._get_container_metrics = AsyncMock(return_value=None)

        result = await autoscaler._compute_target_containers_internal(current_replicas=3)
        assert result == 3
