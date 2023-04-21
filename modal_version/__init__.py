# Copyright Modal Labs 2022
"""Specifies the `modal.__version__` number for the client package."""

from ._version_generated import build_number  # Written by GitHub

# As long as we're on 0.*, all versions are published automatically
major_number = 0

# Bump this manually on any major changes
minor_number = 49

# Right now, set the patch number (the 3rd field) to the job run number in GitHub
__version__ = f"{major_number}.{minor_number}.{build_number}"
