# Copyright Modal Labs 2023
from datetime import date

from modal_utils.async_utils import synchronize_api

from .exception import deprecation_error


class _SharedVolume:
    def __init__(self, *args, **kwargs):
        """`SharedVolume(...)` is deprecated. Please use `NetworkFileSystem.new(...)` instead."""
        deprecation_error(date(2023, 7, 5), _SharedVolume.__init__.__doc__)

    @staticmethod
    def new(*args, **kwargs):
        """`SharedVolume.new(...)` is deprecated. Please use `NetworkFileSystem.new(...)` instead."""
        deprecation_error(date(2023, 7, 5), _SharedVolume.new.__doc__)

    @staticmethod
    def persisted(*args, **kwargs):
        """`SharedVolume.persisted(...)` is deprecated. Please use `NetworkFileSystem.persisted(...)` instead."""
        deprecation_error(date(2023, 7, 5), _SharedVolume.persisted.__doc__)


SharedVolume = synchronize_api(_SharedVolume)
