# Copyright Modal Labs 2025
# pyright: reportMissingImports=false
import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from modal.experimental.flash import (
    MAX_FAILURES,
    _FlashManager,
    _FlashPrometheusAutoscaler,
)


class TestFlashInternalMetricAutoscalerLogic:
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
            (
                "cpu_usage_percent",
                0.5,
                [0.1, 0.05, 0.05],
                2,
                1,
            ),  # Low CPU single container
            (
                "cpu_usage_percent",
                0.5,
                [0.2],
                3,
                3,
            ),  # Low CPU single container and other containers unhealthy
            (
                "cpu_usage_percent",
                0.5,
                [0.6, 0.7, 0.8, 0.9],
                5,
                8,
            ),  # High CPU multiple containers
            ("memory_usage_percent", 0.6, [0.9], 2, 3),  # High memory usage
            (
                "memory_usage_percent",
                0.6,
                [0.52],
                2,
                2,
            ),  # Low memory usage within tolerance
        ],
    )
    async def test_metric_scaling(
        self,
        autoscaler,
        flash_metric,
        target_value,
        metrics,
        current_replicas,
        expected_replicas,
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


class TestFlashManagerStopping:
    @pytest.fixture
    def mock_tunnel_manager(self):
        """Mock the tunnel manager async context manager."""
        mock_tunnel_manager = MagicMock()
        mock_tunnel = MagicMock()
        mock_tunnel.url = "https://test.modal.test"
        mock_tunnel_manager.__aenter__ = AsyncMock(return_value=mock_tunnel)
        mock_tunnel_manager.__aexit__ = AsyncMock()
        return mock_tunnel_manager

    @pytest.fixture
    def flash_manager(self, client, mock_tunnel_manager):
        """Create a FlashManager with mocked dependencies."""
        with (
            patch.dict(os.environ, {"MODAL_TASK_ID": "test-task-123"}),
            patch(
                "modal.experimental.flash._forward_tunnel",
                return_value=mock_tunnel_manager,
            ),
        ):
            manager = _FlashManager(client=client, port=8000)
            return manager

    @pytest.mark.asyncio
    async def test_heartbeat_failure_increments_counter(self, flash_manager):
        """Test that heartbeat failures properly increment the failure counter."""

        flash_manager.tunnel = MagicMock()
        flash_manager.tunnel.url = "https://test.modal.test"
        flash_manager.client.stub.FlashContainerRegister = AsyncMock()
        flash_manager.client.stub.FlashContainerDeregister = AsyncMock()
        flash_manager.is_port_connection_healthy = AsyncMock(
            return_value=(False, Exception("Persistent network error"))
        )

        heartbeat_task = asyncio.create_task(flash_manager._run_heartbeat("test.modal.test", 443))
        await asyncio.sleep(1)
        try:
            heartbeat_task.cancel()
            await heartbeat_task
        except asyncio.CancelledError:
            pass  # Expected when task is cancelled

        # Check that failures were recorded
        assert flash_manager.num_failures > 0

    @pytest.mark.asyncio
    async def test_heartbeat_triggers_failure(self, flash_manager):
        """Test that heartbeat failures properly increment the failure counter."""

        flash_manager.tunnel = MagicMock()
        flash_manager.client.stub.FlashContainerRegister = AsyncMock()
        flash_manager.client.stub.FlashContainerDeregister = AsyncMock()
        heartbeat_task = asyncio.create_task(flash_manager._run_heartbeat("test.modal.test", 443))
        drain_task = asyncio.create_task(flash_manager._drain_container())

        with patch.object(flash_manager, "stop", new_callable=AsyncMock) as mock_stop:
            for i in range(3, 5):
                flash_manager.num_failures = i
                await asyncio.sleep(1)
                if i <= MAX_FAILURES:
                    assert flash_manager.num_failures == i
                    assert mock_stop.call_count == 0
                else:
                    assert flash_manager.num_failures == i
                    assert mock_stop.call_count == 1

        try:
            heartbeat_task.cancel()
            drain_task.cancel()
            await heartbeat_task
            await drain_task
        except asyncio.CancelledError:
            pass  # Expected when task is cancelled

        # Check that failures were recorded
        assert flash_manager.num_failures > 0

    @pytest.mark.asyncio
    async def test_full_failure_and_stop_integration(self, flash_manager):
        """Test the full integration: failures -> drain -> stop."""

        with (
            patch.object(flash_manager, "stop", new_callable=AsyncMock) as mock_stop,
        ):
            # Set up mocks
            flash_manager.tunnel = MagicMock()
            flash_manager.tunnel.url = "https://test.modal.test"

            # Mock HTTP client to always fail
            flash_manager.is_port_connection_healthy = AsyncMock(
                return_value=(False, Exception("Persistent network error"))
            )

            # Mock client stub methods
            flash_manager.client.stub.FlashContainerRegister = AsyncMock()
            flash_manager.client.stub.FlashContainerDeregister = AsyncMock()
            flash_manager.client.stub.ContainerStop = AsyncMock()

            flash_manager.num_failures = 3

            # Start both background tasks
            heartbeat_task = asyncio.create_task(flash_manager._run_heartbeat("test.modal.test", 443))
            drain_task = asyncio.create_task(flash_manager._drain_container())

            await asyncio.sleep(1)

            heartbeat_task.cancel()
            drain_task.cancel()

            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

            try:
                await drain_task
            except asyncio.CancelledError:
                pass

            assert flash_manager.num_failures > 3, f"Expected > 3 failures, got {flash_manager.num_failures}"

            mock_stop.assert_called_once()
