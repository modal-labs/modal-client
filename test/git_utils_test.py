# Copyright Modal Labs 2025
import asyncio
import pytest
import sys
from unittest import mock

from modal._utils.git_utils import get_git_commit_info, run_command_fallible


@mock.patch("asyncio.create_subprocess_exec")
@pytest.mark.asyncio
async def test_run_command_fallible_success_mocked(mock_subprocess):
    mock_process = mock.AsyncMock()
    mock_process.communicate.return_value = (b"test output", b"")
    mock_process.returncode = 0
    mock_subprocess.return_value = mock_process

    result = await run_command_fallible(["git", "status"])

    assert result == "test output"
    mock_subprocess.assert_called_once_with(
        "git", "status", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )


@mock.patch("asyncio.create_subprocess_exec")
@pytest.mark.asyncio
async def test_run_command_fallible_failure(mock_subprocess):
    mock_process = mock.AsyncMock()
    mock_process.communicate.return_value = (b"", b"error message")
    mock_process.returncode = 1
    mock_subprocess.return_value = mock_process

    result = await run_command_fallible(["git", "status"])

    assert result is None


@mock.patch("asyncio.create_subprocess_exec")
@pytest.mark.asyncio
async def test_run_command_fallible_exception(mock_subprocess):
    mock_subprocess.side_effect = Exception("Command failed")

    result = await run_command_fallible(["git", "status"])

    assert result is None


@pytest.mark.skipif(sys.platform == "win32", reason="Skipping on Windows")
@pytest.mark.asyncio
async def test_run_command_fallible_success_real():
    result = await run_command_fallible(["echo", "hello world"])

    assert result == "hello world"


@pytest.mark.asyncio
async def test_run_command_fallible_unknown_command_real():
    result = await run_command_fallible(["command_that_does_not_exist"])

    assert result is None


def assert_commit_info(result, expected_values):
    """Helper function to assert CommitInfo fields match expected values."""
    assert result is not None
    for field, expected in expected_values.items():
        assert getattr(result, field) == expected, (
            f"Field {field} mismatch: expected {expected}, got {getattr(result, field)}"
        )


@mock.patch("modal._utils.git_utils.run_command_fallible")
@pytest.mark.asyncio
async def test_get_git_commit_info_success(mock_run_command):
    mock_run_command.side_effect = [
        "abc123\n1609459200\nTest User\ntest@example.com",  # git log
        "main",  # git branch
        "",  # git status (clean)
        "git@github.com:modal-labs/modal-client.git",  # git remote
    ]

    result = await get_git_commit_info()

    expected = {
        "vcs": "git",
        "commit_hash": "abc123",
        "commit_timestamp": 1609459200,
        "author_name": "Test User",
        "author_email": "test@example.com",
        "branch": "main",
        "dirty": False,
        "repo_url": "git@github.com:modal-labs/modal-client.git",
    }
    assert_commit_info(result, expected)


@mock.patch("modal._utils.git_utils.run_command_fallible")
@pytest.mark.asyncio
async def test_get_git_commit_info_dirty_repo(mock_run_command):
    mock_run_command.side_effect = [
        "abc123\n1609459200\nTest User\ntest@example.com",
        "main",
        "?? main.py",  # Modified file indicates dirty repo
        "https://github.com/modal-labs/modal-client.git",
    ]

    result = await get_git_commit_info()

    expected = {
        "vcs": "git",
        "commit_hash": "abc123",
        "commit_timestamp": 1609459200,
        "author_name": "Test User",
        "author_email": "test@example.com",
        "branch": "main",
        "dirty": True,
        "repo_url": "https://github.com/modal-labs/modal-client.git",
    }
    assert_commit_info(result, expected)


@mock.patch("modal._utils.git_utils.run_command_fallible")
@pytest.mark.asyncio
async def test_get_git_commit_info_missing_remote(mock_run_command):
    mock_run_command.side_effect = [
        "abc123\n1609459200\nTest User\ntest@example.com",
        "main",
        "",
        None,  # git remote fails with "error: No such remote 'origin'"
    ]

    result = await get_git_commit_info()

    expected = {
        "vcs": "git",
        "commit_hash": "abc123",
        "commit_timestamp": 1609459200,
        "author_name": "Test User",
        "author_email": "test@example.com",
        "branch": "main",
        "dirty": False,
        "repo_url": "",
    }
    assert_commit_info(result, expected)


@mock.patch("modal._utils.git_utils.run_command_fallible")
@pytest.mark.asyncio
async def test_get_git_commit_info_missing_author_email(mock_run_command):
    mock_run_command.side_effect = [
        "abc123\n1609459200\nTest User\n",  # Missing author email
        "main",
        "",
        "https://github.com/modal-labs/modal-client.git",
    ]

    result = await get_git_commit_info()

    expected = {
        "vcs": "git",
        "commit_hash": "abc123",
        "commit_timestamp": 1609459200,
        "author_name": "Test User",
        "author_email": "",
        "branch": "main",
        "dirty": False,
        "repo_url": "https://github.com/modal-labs/modal-client.git",
    }
    assert_commit_info(result, expected)


@mock.patch("modal._utils.git_utils.run_command_fallible")
@pytest.mark.asyncio
async def test_get_git_commit_info_new_repo(mock_run_command):
    # Responses for a new repository with no commits
    mock_run_command.side_effect = [
        None,  # git log fails with "fatal: ambiguous argument 'HEAD'..."
        None,  # git rev-parse fails with "fatal: ambiguous argument 'HEAD'..."
        "?? main.py",  # Modified file
        None,  # git remote fails with "error: No such remote 'origin'"
    ]

    result = await get_git_commit_info()

    assert result is None
