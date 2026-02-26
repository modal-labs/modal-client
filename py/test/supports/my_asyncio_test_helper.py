# Copyright Modal Labs 2025
"""
Helper module to test that files with 'asyncio' in the name are not incorrectly filtered.
"""

from modal._utils.async_utils import _extract_user_call_frame


def call_extract_from_asyncio_named_file():
    """Function that calls _extract_user_call_frame from a file with 'asyncio' in its name."""
    return _extract_user_call_frame()
