# Copyright Modal Labs 2025
import os
import subprocess
from typing import Dict, List, Optional, Union


def run_command(args: List[str]) -> Optional[str]:
    """Run a command and return its output.

    Args:
        args: Command and arguments as a list

    Returns:
        Command output as string or None if command failed
    """
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_commit_info() -> Dict[str, Union[str, bool, int]]:
    """Collect git information about the current repository."""

    # Check if we're in a git repository
    if not os.path.exists(".git") and run_command(["git", "rev-parse", "--is-inside-work-tree"]) != "true":
        return {}

    git_info = {
        "vcs": "git",
    }

    # Get branch name
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        git_info["branch"] = branch
    else:
        # If there's an error getting the branch name, bail
        return {}

    # Get commit hash
    commit_hash = run_command(["git", "rev-parse", "HEAD"])
    if commit_hash:
        git_info["commit_hash"] = commit_hash
    else:
        # If there's an error getting the commit hash, bail
        return {}

    # Check if working directory is dirty
    dirty_output = run_command(["git", "status", "--porcelain"])
    git_info["dirty"] = bool(dirty_output)

    # Get repository URL
    repo_url = run_command(["git", "remote", "get-url", "origin"])
    if repo_url:
        git_info["repo_url"] = repo_url

    # Get author name
    author_name = run_command(["git", "show", "-s", "--format=%an", "HEAD"])
    if author_name:
        git_info["author_name"] = author_name

    # Get author email
    author_email = run_command(["git", "show", "-s", "--format=%ae", "HEAD"])
    if author_email:
        git_info["author_email"] = author_email

    return git_info
