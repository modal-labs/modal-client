# Copyright Modal Labs 2025
import os
import pytest
import subprocess
from unittest.mock import MagicMock, patch

from modal.experimental.flash import _FlashManager


class TestFlashManagerLsofCodePath:
    """Test cases specifically designed to hit the lsof code path in check_process_is_running."""

    @pytest.fixture
    def flash_manager(self, client):
        """Create a FlashManager for testing."""
        with patch.dict(os.environ, {"MODAL_TASK_ID": "test-task-123"}):
            manager = _FlashManager(client=client, port=8000)
            return manager

    def test_process_none_lsof_finds_process(self, flash_manager):
        """Test case: process=None, lsof succeeds and finds a process on the port."""
        # Mock subprocess.run to simulate lsof finding a process
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "python3   12345 user   10u  IPv4 123456      0t0  TCP *:8000 (LISTEN)"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            running, error = flash_manager.check_process_is_running(process=None)

            # Verify lsof was called with correct arguments
            mock_run.assert_called_once_with(
                ["lsof", "-i:8000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Should return True since lsof found a process
            assert running is True
            assert error is None

    def test_process_none_lsof_no_output(self, flash_manager):
        """Test case: process=None, lsof succeeds but finds no process on the port."""
        # Mock subprocess.run to simulate lsof finding nothing
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            running, error = flash_manager.check_process_is_running(process=None)

            # Verify lsof was called
            mock_run.assert_called_once_with(
                ["lsof", "-i:8000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Should return True (fallback case when lsof succeeds but no output)
            assert running is True
            assert error is None

    def test_process_none_lsof_nonzero_exit(self, flash_manager):
        """Test case: process=None, lsof exits with non-zero code."""
        # Mock subprocess.run to simulate lsof failing (non-zero exit code)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            running, error = flash_manager.check_process_is_running(process=None)

            # Verify lsof was called
            mock_run.assert_called_once_with(
                ["lsof", "-i:8000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Should return True (fallback case when lsof fails)
            assert running is True
            assert error is None

    def test_process_none_lsof_exception(self, flash_manager):
        """Test case: process=None, lsof raises an exception."""
        # Mock subprocess.run to raise an exception
        with patch("subprocess.run", side_effect=FileNotFoundError("lsof command not found")) as mock_run:
            running, error = flash_manager.check_process_is_running(process=None)

            # Verify lsof was called
            mock_run.assert_called_once_with(
                ["lsof", "-i:8000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Should return False with error when lsof throws exception
            assert running is False
            assert error is not None
            assert "Error checking port 8000 with lsof" in str(error)
            assert "lsof command not found" in str(error)

    def test_running_process_lsof_finds_process(self, flash_manager):
        """Test case: process is running (poll() returns None), lsof finds a process."""
        # Mock a running process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is still running
        mock_process.pid = 12345

        # Mock subprocess.run to simulate lsof finding a process
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "python3   12345 user   10u  IPv4 123456      0t0  TCP *:8000 (LISTEN)"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            running, error = flash_manager.check_process_is_running(process=mock_process)

            # Verify process.poll() was called
            mock_process.poll.assert_called_once()

            # Verify lsof was called since process is still running
            mock_run.assert_called_once_with(
                ["lsof", "-i:8000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Should return True since lsof found a process
            assert running is True
            assert error is None

    def test_exited_process_no_lsof_call(self, flash_manager):
        """Test case: process has exited (poll() returns exit code), should not call lsof."""
        # Mock an exited process
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process has exited with code 1
        mock_process.pid = 12345

        with patch("subprocess.run") as mock_run:
            running, error = flash_manager.check_process_is_running(process=mock_process)

            # Verify process.poll() was called
            mock_process.poll.assert_called_once()

            # Verify lsof was NOT called since process has exited
            mock_run.assert_not_called()

            # Should return False with error since process has exited
            assert running is False
            assert error is not None
            assert f"Process {mock_process.pid} exited with code 1" in str(error)

    def test_running_process_lsof_exception(self, flash_manager):
        """Test case: process is running, but lsof raises an exception."""
        # Mock a running process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is still running

        # Mock subprocess.run to raise an exception
        with patch("subprocess.run", side_effect=OSError("Permission denied")) as mock_run:
            running, error = flash_manager.check_process_is_running(process=mock_process)

            # Verify process.poll() was called
            mock_process.poll.assert_called_once()

            # Verify lsof was called
            mock_run.assert_called_once()

            # Should return False with error when lsof throws exception
            assert running is False
            assert error is not None
            assert "Error checking port 8000 with lsof" in str(error)
            assert "Permission denied" in str(error)

    def test_different_ports_lsof_call(self, flash_manager_with_port):
        """Test that lsof is called with the correct port number for different managers."""
        # Create manager with different port
        with patch.dict(os.environ, {"MODAL_TASK_ID": "test-task-456"}):
            manager = _FlashManager(client=flash_manager_with_port, port=9000)

        # Mock subprocess.run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "some output"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            manager.check_process_is_running(process=None)

            # Verify lsof was called with the correct port
            mock_run.assert_called_once_with(
                ["lsof", "-i:9000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

    @pytest.fixture
    def flash_manager_with_port(self, client):
        """Helper fixture to create managers with different ports."""
        return client
