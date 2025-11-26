# Copyright Modal Labs 2025
# pyright: reportMissingImports=false
import asyncio
import os
import pytest
from types import MethodType
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

from modal.experimental.flash import _FlashManager, _FlashPrometheusAutoscaler


@pytest.fixture
def mock_tunnel_manager():
    """Mock the tunnel manager async context manager."""
    mock_tunnel_manager = MagicMock()
    mock_tunnel = MagicMock()
    mock_tunnel.url = "https://test.modal.test"
    mock_tunnel_manager.__aenter__ = AsyncMock(return_value=mock_tunnel)
    mock_tunnel_manager.__aexit__ = AsyncMock()
    return mock_tunnel_manager


@pytest.fixture
def flash_manager(client, mock_tunnel_manager):
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


class _DummyContainer:
    def __init__(self, host: str):
        self.host = host
        self.port = 443
        self.task_id = f"task-{host}"  # Add missing task_id attribute


class _DummySample:
    def __init__(self, value: float):
        self.value = value


class TestFlashAutoscalerLogic:
    @pytest.fixture
    def autoscaler(self, client):
        with patch("aiohttp.ClientSession"):
            return _FlashPrometheusAutoscaler(
                client=client,
                app_name="test_app",
                cls_name="test_cls",
                metrics_endpoint="metrics",
                target_metric="test_metric",
                target_metric_value=10,
                min_containers=None,
                max_containers=None,
                scale_up_tolerance=0.1,
                scale_down_tolerance=0.1,
                scale_up_stabilization_window_seconds=0,
                scale_down_stabilization_window_seconds=300,
                autoscaling_interval_seconds=15,
                buffer_containers=None,
            )

    def _make_autoscaler(self, autoscaler: _FlashPrometheusAutoscaler, metrics_by_host: dict[str, float]):
        async def _get_all_containers(_self):
            return [_DummyContainer(h) for h in metrics_by_host.keys()]

        async def _get_metrics(_self, url: str):
            host = urlparse(url).hostname or ""
            value = metrics_by_host.get(host, None)
            if value is None:
                return None
            return {"test_metric": [_DummySample(value)]}

        autoscaler._get_all_containers = MethodType(_get_all_containers, autoscaler)  # type: ignore
        autoscaler._get_metrics = MethodType(_get_metrics, autoscaler)  # type: ignore
        return autoscaler

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metrics_by_host,current_replicas,overprovision_containers,expected_replicas",
        [
            # No containers discovered; current_replicas == 0 should return 1
            ({}, 0, None, 1),
            # 3 containers, all emit value 15 with target 10 -> avg=15 -> ratio=1.5 -> ceil(3*1.5)=5
            ({"h1": 15.0, "h2": 15.0, "h3": 15.0}, 3, None, 5),
            # 4 containers, all emit value 5 with target 10 -> avg=5 -> ratio=0.5 -> ceil(4*0.5)=2
            ({"h1": 5.0, "h2": 5.0, "h3": 5.0, "h4": 5.0}, 4, None, 2),
            # 3 containers, only one emits 10 (target), two unhealthy -> both assumed at target for scale-up calc
            ({"h1": 10.0, "h2": None, "h3": None}, 3, None, 5),
            # current_replicas (1) < discoverable containers (3) -> adjusted up to 3; metrics equal target -> stay at 3
            ({"h1": 10.0, "h2": 10.0, "h3": 10.0}, 1, None, 3),
            # Overprovision reduces denominator in scale-up avg: 3 containers at 15,
            # overprov=1 -> (45)/(3-1)=22.5 -> ratio=2.25 -> ceil((3-1)*2.25)=5
            ({"h1": 15.0, "h2": 15.0, "h3": 15.0}, 3, 1, 5),
            # All below target triggers scale-down; overprovision should not change scale-down computation
            ({"h1": 5.0, "h2": 5.0, "h3": 5.0, "h4": 5.0}, 4, 2, 2),
            # Overprovision greater than discoverable containers -> denominator would be <= 0 -> floored to 1
            ({"h1": 15.0}, 1, 5, 2),
            # One healthy at target, one unhealthy; with overprovision=1 denominator becomes 1, leading to scale-up.
            ({"h1": 10.0, "h2": None}, 2, 1, 3),
            # 5 containers, h2 and h4 unhealthy, others at 15.0
            ({"h1": 15.0, "h2": None, "h3": 15.0, "h4": None, "h5": 15.0}, 5, None, 8),
            # 5 containers, mixed metrics and some unhealthy
            ({"h1": 12.0, "h2": 8.0, "h3": 15.0, "h4": None, "h5": None}, 5, None, 7),
            # 4 containers all with the same metric value
            ({"h1": 5.0, "h2": 5.0, "h3": 5.0, "h4": 5.0}, 4, None, 2),
            # Custom target values, tolerances, and overprovision settings
            ({"h1": 25.0, "h2": 30.0, "h3": None}, 3, 1, 8),
            # All unhealthy, 20 current replicas, 17 provisioned, target_metric_value=20.0
            (
                {f"h{i}": None for i in range(1, 21)},
                20,
                1,
                33,
            ),
            # 13 containers, 0 healthy, 9 unhealthy, 1 missing, 3 buffer
            # expected number should just be > 10
            (
                {f"h{i}": None for i in range(1, 10)},
                13,
                3,
                15,
            ),
        ],
    )
    async def test_metric_scaling(
        self,
        metrics_by_host,
        current_replicas,
        overprovision_containers,
        expected_replicas,
        autoscaler,
    ):
        autoscaler = self._make_autoscaler(autoscaler, metrics_by_host)

        if overprovision_containers is not None:
            autoscaler.buffer_containers = overprovision_containers

        result = await autoscaler._compute_target_containers(current_replicas=current_replicas)
        assert result == expected_replicas


_MAX_FAILURES = 10


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
                buffer_containers=None,
            )


class TestFlashManagerStopping:
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
    async def test_heartbeat_success_resets_counter(self, flash_manager):
        """Test that heartbeat failures properly increment the failure counter."""

        flash_manager.tunnel = MagicMock()
        flash_manager.tunnel.url = "https://test.modal.test"
        flash_manager.client.stub.FlashContainerRegister = AsyncMock()
        flash_manager.client.stub.FlashContainerDeregister = AsyncMock()
        flash_manager.is_port_connection_healthy = AsyncMock(return_value=(True, None))

        heartbeat_task = asyncio.create_task(flash_manager._run_heartbeat("test.modal.test", 443))
        await asyncio.sleep(1)
        try:
            heartbeat_task.cancel()
            await heartbeat_task
        except asyncio.CancelledError:
            pass  # Expected when task is cancelled

        # Check that failures were recorded
        assert flash_manager.num_failures == 0

    @pytest.mark.asyncio
    async def test_heartbeat_triggers_failure(self, flash_manager):
        """Test that heartbeat failures properly increment the failure counter."""

        flash_manager.tunnel = MagicMock()
        flash_manager.client.stub.FlashContainerRegister = AsyncMock()
        flash_manager.client.stub.FlashContainerDeregister = AsyncMock()
        heartbeat_task = asyncio.create_task(flash_manager._run_heartbeat("test.modal.test", 443))
        drain_task = asyncio.create_task(flash_manager._drain_container())

        with patch.object(flash_manager, "stop", new_callable=AsyncMock) as mock_stop:
            for i in range(_MAX_FAILURES, _MAX_FAILURES + 2):
                flash_manager.num_failures = i
                if i <= _MAX_FAILURES:
                    assert flash_manager.num_failures == i
                    assert mock_stop.call_count == 0
                else:
                    assert flash_manager.num_failures > _MAX_FAILURES
                await asyncio.sleep(1)
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
    async def test_hearbeat_on_dead_process(self, flash_manager, servicer):
        """Test that flash heartbeat kills task if process dies."""
        flash_manager.tunnel = MagicMock()
        flash_manager.tunnel.url = "https://test.modal.test"
        # Set a short startup_timeout so failures are recorded after timeout expires
        flash_manager.startup_timeout = 0.05
        # Spin up a dead process, and set is_port_connection_healthy to call the real method on it.
        import subprocess

        # Start a process that will exit immediately
        dead_process = subprocess.Popen(["sleep", "0"])
        dead_process.wait(timeout=2)  # ensure it is dead

        async def mock_is_port_connection_healthy(process, timeout=0.5):
            return await type(flash_manager).is_port_connection_healthy(flash_manager, dead_process, timeout=timeout)

        flash_manager.is_port_connection_healthy = AsyncMock(side_effect=mock_is_port_connection_healthy)
        flash_manager.client.stub.FlashContainerDeregister = AsyncMock()

        host = "heartbeat-host.modal.test"
        port = 9000

        task = asyncio.create_task(flash_manager._run_heartbeat(host, port))
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

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

            flash_manager.num_failures = _MAX_FAILURES

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

            assert flash_manager.num_failures > _MAX_FAILURES, (
                f"Expected > {_MAX_FAILURES} failures, got {flash_manager.num_failures}"
            )

            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_heartbeat(self, flash_manager, servicer):
        """Integration test that _run_heartbeat registers using FlashContainerRegister RPC."""

        flash_manager.tunnel = MagicMock()
        flash_manager.tunnel.url = "https://test.modal.test"
        flash_manager.is_port_connection_healthy = AsyncMock(return_value=(True, None))

        host = "heartbeat-host.modal.test"
        port = 9000

        task = asyncio.create_task(flash_manager._run_heartbeat(host, port))
        await asyncio.sleep(0.2)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(servicer.flash_container_registrations) >= 1

    async def test_flash_startup_heartbeat(self, flash_manager):
        """Test that flash startup heartbeat registers the container."""
        flash_manager.tunnel = MagicMock()
        flash_manager.tunnel.url = "https://test.modal.test"

        async def side_effect(*args, **kwargs):
            if not hasattr(flash_manager, "_call_count"):
                flash_manager._call_count = 0
            flash_manager._call_count += 1
            if flash_manager._call_count <= 30:
                return (False, None)
            return (True, None)

        flash_manager.is_port_connection_healthy = AsyncMock(side_effect=side_effect)

        host = "heartbeat-host.modal.test"
        port = 9000

        task = asyncio.create_task(flash_manager._run_heartbeat(host, port))
        await asyncio.sleep(0.2)
        task.cancel()

    async def test_flash_startup_timeout(self, flash_manager):
        """Test that flash startup timeout works."""
        flash_manager.tunnel = MagicMock()
        flash_manager.tunnel.url = "https://test.modal.test"
        flash_manager.startup_timeout = 0.05
        flash_manager.is_port_connection_healthy = AsyncMock(return_value=(False, None))
        flash_manager.client.stub.FlashContainerDeregister = AsyncMock()

        task = asyncio.create_task(flash_manager._run_heartbeat("heartbeat-host.modal.test", 9000))
        await asyncio.sleep(0.2)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert flash_manager.num_failures > 0
        assert flash_manager.num_failures == _MAX_FAILURES
