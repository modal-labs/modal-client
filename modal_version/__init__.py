# Copyright Modal Labs 2025
"""Supplies the current version of the modal client library."""

__version__ = "1.3.1.dev21"

# For development, we use the mount whose SHA is on the `main` branch and is in the history of the current branch
# If we can not find the mount for the dynamic version, then we'll fallback to static version.
__static_version__ = __version__

import os

if directory := os.getenv("MODAL_DEVELOPMENT_DIRECTORY"):
    import subprocess
    from contextlib import suppress

    with suppress(Exception):
        result = subprocess.run(
            "git log --format='%H' -1 $(git merge-base origin/main HEAD) -- client",
            cwd=directory,
            capture_output=True,
            text=True,
            shell=True,
        )
        sha = result.stdout.strip()
        __version__ = f"{__static_version__}+{sha[:8]}"
