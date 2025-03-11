# Copyright Modal Labs 2025
import asyncio
from typing import Optional

from modal.config import logger
from modal_proto import api_pb2


async def run_command_fallible(args: list[str]) -> Optional[str]:
    try:
        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, _ = await process.communicate()

        if process.returncode != 0:
            logger.debug(f"Command {args} exited with code {process.returncode}")
            return None

        return stdout_bytes.decode("utf-8").strip()

    except Exception as e:
        logger.debug(f"Command {args} failed: {repr(e)}")
        return None


async def get_git_commit_info() -> Optional[api_pb2.CommitInfo]:
    """Collect git information about the current repository asynchronously."""
    git_info: api_pb2.CommitInfo = api_pb2.CommitInfo(vcs="git")

    commands = [
        # Get commit hash, timestamp, author name, and author email
        ["git", "log", "-1", "--format=%H%n%ct%n%an%n%ae", "HEAD"],
        # Get branch name
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        # Check if working directory is dirty
        ["git", "status", "--porcelain"],
        ["git", "remote", "get-url", "origin"],
    ]

    tasks = (run_command_fallible(cmd) for cmd in commands)
    (log_info, branch, status, origin_url) = await asyncio.gather(*tasks)

    if not branch:
        return None

    git_info.branch = branch

    if not log_info:
        return None

    info_lines = log_info.split("\n")
    if len(info_lines) < 4:
        # If we didn't get all expected lines, bail
        logger.debug(f"Log info returned only {len(info_lines)} lines")
        return None

    try:
        git_info.commit_hash = info_lines[0]
        git_info.commit_timestamp = int(info_lines[1])
        git_info.author_name = info_lines[2]
        git_info.author_email = info_lines[3]
    except (ValueError, IndexError):
        logger.debug(f"Failed to parse git log info: {log_info}")
        return None

    git_info.dirty = bool(status)

    if origin_url:
        git_info.repo_url = origin_url

    return git_info
