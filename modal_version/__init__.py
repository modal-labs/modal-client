# Copyright Modal Labs 2022
"""Specifies the `modal.__version__` number for the client package."""

from ._version_generated import build_number

# As long as we're on 0.*, all versions are published automatically
major_number = 0

# Bump this manually on breaking changes, then reset the number in _version_generated.py
minor_number = 61

# Right now, automatically increment the patch number in CI
__version__ = f"{major_number}.{minor_number}.{max(build_number, 0)}"
