# Copyright Modal Labs 2025
import os
import re
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

    # Get commit hash
    commit_hash = run_command(["git", "rev-parse", "HEAD"])
    if commit_hash:
        git_info["commit_hash"] = commit_hash

    # Check if working directory is dirty
    dirty_output = run_command(["git", "status", "--porcelain"])
    git_info["dirty"] = bool(dirty_output)

    # Get repository URL
    repo_url = run_command(["git", "remote", "get-url", "origin"])
    if repo_url:
        git_info["repo_url"] = repo_url

        # Generate commit URL only if we have both repo_url and commit_hash
        if commit_hash and repo_url:
            commit_url = None
            if "github.com" in repo_url:
                base_url = re.sub(r"git@github\.com:", "https://github.com/", repo_url)
                base_url = re.sub(r"\.git$", "", base_url)
                commit_url = f"{base_url}/commit/{commit_hash}"
            elif "gitlab.com" in repo_url:
                base_url = re.sub(r"git@gitlab\.com:", "https://gitlab.com/", repo_url)
                base_url = re.sub(r"\.git$", "", base_url)
                commit_url = f"{base_url}/-/commit/{commit_hash}"
            elif "bitbucket.org" in repo_url:
                base_url = re.sub(r"git@bitbucket\.org:", "https://bitbucket.org/", repo_url)
                base_url = re.sub(r"\.git$", "", base_url)
                commit_url = f"{base_url}/commits/{commit_hash}"

            if commit_url:
                git_info["commit_url"] = commit_url

    return git_info
